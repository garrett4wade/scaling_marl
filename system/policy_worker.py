import multiprocessing
import signal
import time
from queue import Empty

from algorithms.utils.util import check
import numpy as np
import psutil
import torch
from torch.multiprocessing import Process as TorchProcess

from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

from utils.timing import Timing
from utils.utils import log, join_or_kill, cuda_envvars_for_policy, memory_stats, TaskType
from algorithms.storage_registries import to_numpy_type
import zmq
from collections import OrderedDict, deque


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class PolicyWorker:
    def __init__(self, worker_idx, cfg, obs_space, share_obs_space, action_space, buffer, policy_queue, actor_queues,
                 report_queue, task_queue, policy_lock, resume_experience_collection_cv, act_shms, act_semaphores,
                 envstep_output_shms, policy_worker_ready_event):
        log.info('Initializing policy worker %d', worker_idx)

        self.worker_idx = worker_idx
        assert len(cfg.policy_worker_gpu_ranks) == 1 or len(cfg.policy_worker_gpu_ranks) == cfg.num_policy_workers, (
            'policy worker gpu ranks must be a list of length 1 or '
            'have the same length as num_policy_workers')
        if len(cfg.policy_worker_gpu_ranks) == 1:
            self.gpu_rank = cfg.policy_worker_gpu_ranks[0]
        else:
            self.gpu_rank = cfg.policy_worker_gpu_ranks[worker_idx]
        self.cfg = cfg
        self.tpdv = dict(device=torch.device(0), dtype=torch.float32)

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space

        self.num_agents = cfg.num_agents
        self.envs_per_actor = cfg.envs_per_actor
        self.num_splits = cfg.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.envs_per_split = self.envs_per_actor // self.num_splits

        self.num_actors_per_group = self.cfg.num_actors // self.cfg.num_actor_groups
        self.envs_per_group = self.num_actors_per_group * self.envs_per_split

        self.device = None
        self.actor_critic = None
        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.model_weights_socket = None

        self.policy_queue = policy_queue
        self.actor_queues = actor_queues
        self.report_queue = report_queue
        self.policy_worker_ready_event = policy_worker_ready_event

        self.request_clients = []

        self.act_shms = act_shms
        self.act_semaphores = act_semaphores

        self.envstep_output_shms = envstep_output_shms

        # queue other components use to talk to this particular worker
        self.task_queue = task_queue

        self.initialized = False
        self.terminate = False
        self.initialized_event = multiprocessing.Event()
        self.initialized_event.clear()

        self.buffer = buffer
        self.model_weights_registries = OrderedDict()
        # TODO: stop experience collection signal

        self.latest_policy_version = -1
        self.num_policy_updates = 0

        self.total_num_samples = 0
        self.total_inference_steps = 0

        self.process = TorchProcess(target=self._run, daemon=True)

    def _init(self, timing):
        with timing.timeit('init'):
            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d...', self.worker_idx)

            torch.set_num_threads(1)

            if self.cfg.cuda:
                # we should already see only one CUDA device, because of env vars
                assert torch.cuda.device_count() == 1
                self.device = torch.device('cuda', index=0)
            else:
                self.device = torch.device('cpu')

            self.rollout_policy = Policy(self.gpu_rank,
                                         self.cfg,
                                         self.obs_space,
                                         self.share_obs_space,
                                         self.action_space,
                                         is_training=False)
            self.rollout_policy.eval_mode()

            worker_nodes_per_learner = len(self.cfg.seg_addrs) // len(self.cfg.model_weights_addrs)
            learner_node_idx = self.cfg.worker_node_idx // worker_nodes_per_learner
            self.model_weights_socket = zmq.Context().socket(zmq.SUB)
            self.model_weights_socket.connect(self.cfg.model_weights_addrs[learner_node_idx])
            self.model_weights_socket.setsockopt(zmq.SUBSCRIBE, b'param')

            for k, v in self.rollout_policy.state_dict().items():
                self.model_weights_registries[k] = (v.shape, to_numpy_type(v.dtype))

            self.policy_worker_ready_event.set()
            # self._update_weights(timing, block=True)
            log.info('Initialized model on the policy worker %d!', self.worker_idx)

        log.info('Policy worker %d initialized', self.worker_idx)
        self.initialized = True
        self.initialized_event.set()

    def _handle_policy_steps(self, timing):
        with torch.no_grad():
            with timing.add_time('inference/prepare_policy_inputs'):
                organized_requests = [(client // self.cfg.num_actor_groups, client % self.cfg.num_actor_groups) for client in self.request_clients]
                envstep_outputs = {}
                policy_inputs = {}
                for k, shm_pairs in self.envstep_output_shms.items():
                    data = np.stack([shm_pairs[group_idx][split_idx] for split_idx, group_idx in organized_requests], 0)
                    envstep_outputs[k] = data
                    if k in self.buffer.policy_input_keys:
                        policy_inputs[k] = data
                # TODO: deal with non-sharing senarios
                policy_inputs['masks'] = np.zeros_like(envstep_outputs['dones'])
                # masks/dones has shape (num_requests, envs_per_split, num_agents, 1)
                policy_inputs['masks'][:] = 1 - np.all(envstep_outputs['dones'], axis=2, keepdims=True)

            with timing.add_time('inference/preprosessing'):
                shared = policy_inputs['obs'].shape[:3] == (len(organized_requests), self.envs_per_group, self.num_agents)
                rollout_bs = len(organized_requests) * self.envs_per_group
                if shared:
                    # all agents simultaneously advance an environment step, e.g. SMAC and MPE
                    for k, v in policy_inputs.items():
                        policy_inputs[k] = v.reshape(rollout_bs * self.num_agents, *v.shape[3:])
                else:
                    # agent advances an environment step in turn, e.g. card games
                    for k, v in policy_inputs.items():
                        policy_inputs[k] = v.reshape(rollout_bs, *v.shape[2:])

            with timing.add_time('inference/to_device'):
                for k, v in policy_inputs.items():
                    policy_inputs[k] = check(v).to(**self.tpdv, non_blocking=True)

            with timing.add_time('inference/inference_step'):
                policy_outputs = self.rollout_policy.get_actions(**policy_inputs)

            with timing.add_time('inference/to_cpu_and_postprosessing'):
                if shared:
                    # all agents simultaneously advance an environment step, e.g. SMAC and MPE
                    for k, v in policy_outputs.items():
                        policy_outputs[k] = _t2n(v).reshape(len(self.request_clients), self.envs_per_group, self.num_agents, *v.shape[1:])
                else:
                    # agent advances an environment step in turn, e.g. card games
                    for k, v in policy_outputs.items():
                        policy_outputs[k] = _t2n(v).reshape(len(self.request_clients), self.envs_per_group, *v.shape[1:])

            with timing.add_time('inference/copy_actions'):
                for i, (split_idx, group_idx) in enumerate(organized_requests):
                    for local_actor_idx in range(self.num_actors_per_group):
                        global_actor_idx = self.num_actors_per_group * group_idx + local_actor_idx
                        env_slice = slice(local_actor_idx * self.envs_per_split, (local_actor_idx + 1) * self.envs_per_split)
                        self.act_shms[global_actor_idx][split_idx][:] = policy_outputs['actions'][i, env_slice]
                        self.act_semaphores[global_actor_idx][split_idx].release()

            with timing.add_time('inference/insert_after_inference'):
                # copy rnn states into small shared memory block
                for k, shm_pairs in self.envstep_output_shms.items():
                    if 'rnn_states' in k:
                        for i, (split_idx, group_idx) in enumerate(organized_requests):
                            shm_pairs[group_idx][split_idx][:] = policy_outputs[k][i]

                insert_data = {**envstep_outputs, **policy_outputs}
                self.buffer.insert(organized_requests, **insert_data)

        self.request_clients = []
        self.total_num_samples += rollout_bs
        self.total_inference_steps += 1

    def _update_weights(self, timing, block=False):
        msg = None

        if block:
            msg = self.model_weights_socket.recv_multipart(flags=0)
        else:
            while True:
                # receive the latest model parameters
                try:
                    msg = self.model_weights_socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

        if msg is None:
            return

        with timing.add_time('update_weights/processing_msg'):
            # msg is multiple (key, tensor) pairs + policy version
            assert len(msg) % 2 == 0
            msg = msg[1:]
            state_dict = OrderedDict()
            for i in range(len(msg) // 2):
                key, value = msg[2 * i].decode('ascii'), msg[2 * i + 1]

                shape, dtype = self.model_weights_registries[key]
                tensor = torch.from_numpy(np.frombuffer(memoryview(value), dtype=dtype).reshape(*shape))

                state_dict[key] = tensor

            learner_policy_version = int(msg[-1].decode('ascii'))

        with timing.time_avg('load_state_dict'), timing.add_time('update_weights/load_state_dict'):
            with self.policy_lock:
                self.rollout_policy.load_state_dict(state_dict)

        self.latest_policy_version = learner_policy_version

        if self.num_policy_updates % 10 == 0:
            log.debug(
                'Updated weights on worker %d, policy_version %d (%s)',
                self.worker_idx,
                self.latest_policy_version,
                str(timing.load_state_dict),
            )
        self.num_policy_updates += 1

    # noinspection PyProtectedMember
    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        psutil.Process().nice(min(self.cfg.default_niceness + 2, 20))

        cuda_envvars_for_policy(self.worker_idx, 'inference')
        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        self._init(timing)

        last_report = last_cache_cleanup = time.time()
        last_report_samples = 0
        cur_split = 0

        # very conservative limit on the minimum number of requests to wait for
        # this will almost guarantee that the system will continue collecting experience
        # at max rate even when 2/3 of workers are stuck for some reason (e.g. doing a long env reset)
        # Although if your workflow involves very lengthy operations that often freeze workers, it can be beneficial
        # to set min_num_requests to 1 (at a cost of potential inefficiency, i.e. policy worker will use very small
        # batches)
        min_num_requests = self.cfg.min_num_requests
        if min_num_requests is None or min_num_requests == -1:
            min_num_requests = self.cfg.num_actor_groups // self.cfg.num_policy_workers
            min_num_requests //= 3
            min_num_requests = max(1, min_num_requests)
        log.info('Min num requests: %d', min_num_requests)

        # Again, very conservative timer. Only wait a little bit, then continue operation.
        wait_for_min_requests = 0.025
        num_requests = deque(maxlen=100)

        while not self.terminate:
            try:
                # TODO: add stop experiment collection signal
                # while self.stop_experience_collection:
                #     with self.resume_experience_collection_cv:
                #         self.resume_experience_collection_cv.wait(timeout=0.05)

                waiting_started = time.time()
                with timing.time_avg('waiting_avg'), timing.add_time('waiting'):
                    while len(self.request_clients) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                        try:
                            policy_requests = self.policy_queue.get_many(timeout=0.005)
                            self.request_clients.extend(policy_requests)
                        except Empty:
                            pass

                if len(self.request_clients) > 0:
                    num_requests.append(len(self.request_clients))

                    with timing.add_time('update_weights'):
                        self._update_weights(timing)

                    with timing.time_avg('inference_avg'), timing.add_time('inference'):
                        self._handle_policy_steps(timing)

                with timing.add_time('get_tasks'):
                    try:
                        task_type, data = self.task_queue.get_nowait()

                        # task from the task_queue
                        if task_type == TaskType.TERMINATE:
                            self.terminate = True
                            break

                        self.task_queue.task_done()
                    except Empty:
                        pass

                with timing.add_time('report'):
                    if time.time() - last_report > 3.0 and 'inference_avg' in timing:
                        timing_stats = dict(wait_policy=timing.waiting_avg, step_policy=timing.inference_avg)
                        samples_since_last_report = self.total_num_samples - last_report_samples

                        stats = memory_stats('policy_worker', self.device)

                        self.report_queue.put(
                            dict(
                                timing=timing_stats,
                                samples=samples_since_last_report,
                                stats=stats,
                            ))
                        last_report = time.time()
                        last_report_samples = self.total_num_samples

                    if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark
                                                                    and self.total_num_samples < 1000):
                        if self.cfg.cuda:
                            torch.cuda.empty_cache()
                        last_cache_cleanup = time.time()

            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on worker %d', self.worker_idx)
                self.terminate = True
            except Exception:
                log.exception('Unknown exception on policy worker')
                self.terminate = True

        self.model_weights_socket.close()
        time.sleep(0.2)
        log.info('Policy worker avg requests: %d, total num sample: %d, total inference steps: %d, timing: %s', np.mean(num_requests), self.total_num_samples,
                 self.total_inference_steps, timing)

    def start_process(self):
        self.process.start()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
