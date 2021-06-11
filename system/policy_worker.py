import multiprocessing
import signal
import time
from collections import deque
from queue import Empty

import numpy as np
import psutil
import torch
from torch.multiprocessing import Process as TorchProcess

from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

from utils.timing import Timing
from utils.utils import log, join_or_kill, cuda_envvars_for_policy, memory_stats, TaskType
import zmq
from collections import OrderedDict


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class PolicyWorker:
    def __init__(self, worker_idx, cfg, obs_space, share_obs_space, action_space, buffer, policy_queue, actor_queues,
                 report_queue, task_queue, policy_lock, resume_experience_collection_cv):
        log.info('Initializing policy worker %d', worker_idx)

        self.worker_idx = worker_idx
        self.cfg = cfg

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space

        self.num_agents = cfg.num_agents
        self.envs_per_actor = cfg.envs_per_actor
        self.num_splits = cfg.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.envs_per_split = self.envs_per_actor // self.num_splits

        self.num_actors_per_policy_worker = self.cfg.num_actors // self.cfg.num_policy_workers
        self.actor_cnt = np.array([0 for _ in range(self.num_splits)])
        self.ready_split_id = -1

        self.device = None
        self.actor_critic = None
        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.model_weights_socket = None

        self.policy_queue = policy_queue
        self.actor_queues = actor_queues
        self.report_queue = report_queue

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

        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('Policy worker %d initialized', self.worker_idx)
        self.initialized = True
        self.initialized_event.set()

    def _handle_policy_steps(self, timing):
        with torch.no_grad():
            with timing.add_time('inference_prepare_policy_inputs'):
                policy_inputs = self.buffer.get_policy_inputs(self.worker_idx, self.ready_split_id)

            with timing.add_time('inference_preprosessing'):
                rollout_bs = self.num_actors_per_policy_worker * self.envs_per_split
                shared = policy_inputs['obs'].shape[:2] == (rollout_bs, self.num_agents)
                if shared:
                    # all agents simultaneously advance an environment step, e.g. SMAC and MPE
                    for k, v in policy_inputs.items():
                        policy_inputs[k] = v.reshape(rollout_bs * self.num_agents, *v.shape[2:])
                else:
                    # agent advances an environment step in turn, e.g. card games
                    for k, v in policy_inputs.items():
                        policy_inputs[k] = v

            with timing.add_time('inference_to_device_and_inference'):
                policy_outputs = self.rollout_policy.get_actions(**policy_inputs)

            with timing.add_time('inference_to_cpu_and_postprosessing'):
                if shared:
                    # all agents simultaneously advance an environment step, e.g. SMAC and MPE
                    for k, v in policy_outputs.items():
                        policy_outputs[k] = _t2n(v).reshape(rollout_bs, self.num_agents, *v.shape[1:])
                else:
                    # agent advances an environment step in turn, e.g. card games
                    for k, v in policy_outputs.items():
                        policy_outputs[k] = _t2n(v)

            with timing.add_time('inference_insert_after_inference'):
                self.buffer.insert_after_inference(self.worker_idx, self.ready_split_id, **policy_outputs)
                for actor_queue in self.actor_queues:
                    actor_queue.put((TaskType.ROLLOUT_STEP, self.ready_split_id))

        self.actor_cnt[self.ready_split_id] = 0
        self.ready_split_id = -1
        self.total_num_samples += rollout_bs

    def _update_weights(self, timing, block=False):
        flags = 0 if block else zmq.NOBLOCK

        try:
            msg = self.model_weights_socket.recv_multipart(flags=flags)
        except zmq.ZMQError:
            return

        # msg is multiple (key, tensor) pairs + policy version
        assert len(msg) % 2 == 1
        state_dict = OrderedDict()
        for i in range(len(msg) // 2):
            key, value = msg[2 * i].decode('ascii'), msg[2 * i + 1]

            dtype, shape = self.model_weights_registries[key]
            tensor = torch.from_numpy(np.frombuffer(memoryview(value), dtype=dtype).reshape(*shape))

            state_dict[key] = tensor

        learner_policy_version = int(msg[-1].decode('ascii'))

        with timing.timeit('weight_update'):
            with self.policy_lock:
                self.actor_critic.load_state_dict(state_dict)

        self.latest_policy_version = learner_policy_version

        if self.num_policy_updates % 10 == 0:
            log.info(
                'Updated weights on worker %d, policy_version %d (%.5f)',
                self.worker_idx,
                self.latest_policy_version,
                timing.weight_update,
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

            self.rollout_policy = Policy(0,
                                         self.cfg,
                                         self.obs_space,
                                         self.share_obs_space,
                                         self.action_space,
                                         is_training=False)
            self.rollout_policy.eval_mode()

            self.model_weights_socket = zmq.Context().socket(zmq.SUB)
            self.model_weights_socket.connect(self.cfg.model_weights_addr)
            self.model_weights_socket.setsockopt(zmq.SUBSCRIBE, b'')

            for k, v in self.rollout_policy.state_dict().items():
                self.model_weights_registries[k] = (v.shape, v.dtype)

            # TODO: the next line should be uncommented if a learner participates in the system
            # self._update_weights(timing, block=True)
            log.info('Initialized model on the policy worker %d!', self.worker_idx)

        last_report = last_cache_cleanup = time.time()
        last_report_samples = 0

        while not self.terminate:
            try:
                # TODO: add stop experiment collection signal
                # while self.stop_experience_collection:
                #     with self.resume_experience_collection_cv:
                #         self.resume_experience_collection_cv.wait(timeout=0.05)

                waiting_started = time.time()
                with timing.time_avg('waiting_avg'), timing.add_time('waiting'):
                    while np.all(self.actor_cnt < self.num_actors_per_policy_worker):
                        try:
                            policy_requests = self.policy_queue.get_many(timeout=0.005)
                            for client_id in policy_requests:
                                split_id = client_id % self.num_splits
                                self.actor_cnt[split_id] += 1
                                if self.actor_cnt[split_id] >= self.num_actors_per_policy_worker:
                                    self.ready_split_id = split_id
                        except Empty:
                            pass

                with timing.add_time('update_weights'):
                    if self.ready_split_id >= 0:
                        self._update_weights(timing)

                with timing.time_avg('inference_avg'), timing.add_time('inference'):
                    if self.initialized and self.ready_split_id >= 0:
                        self._handle_policy_steps(timing)

                with timing.add_time('extra_jobs'):
                    try:
                        task_type, data = self.task_queue.get_nowait()

                        # task from the task_queue
                        if task_type == TaskType.INIT:
                            self._init()
                        elif task_type == TaskType.TERMINATE:
                            self.terminate = True
                            break
                        elif task_type == TaskType.INIT_MODEL:
                            self._init_model(data)

                        self.task_queue.task_done()
                    except Empty:
                        pass

                    if time.time() - last_report > 3.0 and 'inference_avg' in timing:
                        timing_stats = dict(wait_policy=timing.waiting_avg, step_policy=timing.inference_avg)
                        samples_since_last_report = self.total_num_samples - last_report_samples

                        stats = memory_stats('policy_worker', self.device)

                        self.report_queue.put(dict(
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
        log.info('Policy worker timing: %s', timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
