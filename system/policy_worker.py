import signal
import time
from queue import Empty

from algorithms.utils.util import check
import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process as TorchProcess

from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

from utils.timing import Timing
from utils.utils import log, join_or_kill, cuda_envvars_for_policy, memory_stats, TaskType
from collections import deque


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class PolicyWorkerPhase:
    WORKING, CLOSING, STOP = range(3)


class PolicyWorker:
    def __init__(self, cfg, policy_id, task_rank, replicate_rank, obs_space, share_obs_space, action_space, buffer,
                 task_queue, policy_queue, actor_queues, report_queue, act_shms, act_semaphores, envstep_output_shms, local_ps, param_lock,
                 ps_policy_version, ps_ready_event, stop_experience_collection_cnt, stop_experience_collection_cond,):
        self.cfg = cfg

        self.policy_id = policy_id
        self.task_rank = task_rank
        self.replicate_rank = replicate_rank

        # worker idx is the global rank on this worker node
        self.worker_idx = (self.cfg.num_policy_workers * self.cfg.num_policies * self.task_rank +
                           self.policy_id * self.cfg.num_policy_workers + self.replicate_rank)
        log.info('Initializing policy worker %d', self.worker_idx)

        self.tpdv = dict(device=torch.device(0), dtype=torch.float32)

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space

        self.num_agents = len(cfg.policy2agents[str(policy_id)])
        self.agent_idx = cfg.policy2agents[str(policy_id)]
        self.envs_per_actor = cfg.envs_per_actor
        self.num_splits = cfg.num_splits
        self.envs_per_split = self.envs_per_actor // self.num_splits

        self.num_actors = self.cfg.num_actors // self.cfg.actor_group_size
        self.num_actor_groups = self.num_actors // self.cfg.num_tasks_per_node
        self.envs_per_group = self.cfg.actor_group_size * self.envs_per_split

        self.device = None
        self.actor_critic = None

        self.policy_queue = policy_queue
        self.report_queue = report_queue
        self.actor_queues = actor_queues

        self.local_ps = local_ps
        self.param_lock = param_lock
        self.ps_policy_version = ps_policy_version
        assert self.ps_policy_version.is_shared()
        self.ps_ready_event = ps_ready_event

        self.local_policy_version = -1

        self.request_clients = []

        self.act_shms = act_shms
        self.act_semaphores = act_semaphores

        self.envstep_output_shms = envstep_output_shms

        # queue other components use to talk to this particular worker
        self.task_queue = task_queue

        self.initialized = False
        self.terminate = False
        self.initialized_event = mp.Event()

        self.buffer = buffer

        self.total_num_samples = 0
        self.total_inference_steps = 0

        # TODO: stop at first and working after receive the ROLLOUT task
        self.phase = PolicyWorkerPhase.WORKING
        self.stop_experience_collection_cnt = stop_experience_collection_cnt
        self.stop_experience_collection_cond = stop_experience_collection_cond

        self.process = TorchProcess(target=self._run, daemon=True)

    def _init(self, timing):
        with timing.timeit('init'):
            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d...', self.worker_idx)

            torch.set_num_threads(32)

            if self.cfg.cuda:
                # we should already see only one CUDA device, because of env vars
                assert torch.cuda.device_count() == 1
                self.device = torch.device('cuda', index=0)
            else:
                self.device = torch.device('cpu')

            # we've already set CUDA_VISIBLE_DEVICES, thus gpu_rank=0 for all policy workers
            self.rollout_policy = Policy(0,
                                         self.cfg,
                                         self.obs_space,
                                         self.share_obs_space,
                                         self.action_space,
                                         is_training=False)
            self.rollout_policy.eval_mode()

            self.ps_ready_event.wait()
            self.maybe_update_weights(timing)
            log.info('Initialized model on the policy worker %d!', self.worker_idx)

            log.info('Policy worker %d initialized', self.worker_idx)
            self.initialized = True
            self.initialized_event.set()

    def maybe_update_weights(self, timing):
        if self.local_policy_version < self.ps_policy_version and self.phase != PolicyWorkerPhase.STOP:
            with self.param_lock.r_locked():
                with timing.time_avg('update_weights/load_state_dict_once'):
                    self.rollout_policy.load_state_dict(self.local_ps)
                    self.local_policy_version = self.ps_policy_version.item()

            if self.local_policy_version % 20 == 0:
                log.debug('Update policy %d to version %d', self.policy_id, self.local_policy_version)

    def _handle_policy_steps(self, timing):
        with torch.no_grad():
            with timing.add_time('inference/prepare_policy_inputs'):
                organized_requests = [(client // self.num_actor_groups, client % self.num_actor_groups)
                                      for client in self.request_clients]
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
                shared = policy_inputs['obs'].shape[:3] == (len(organized_requests), self.envs_per_group,
                                                            self.num_agents)
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
                        policy_outputs[k] = _t2n(v).reshape(len(self.request_clients), self.envs_per_group,
                                                            self.num_agents, *v.shape[1:])
                else:
                    # agent advances an environment step in turn, e.g. card games
                    for k, v in policy_outputs.items():
                        policy_outputs[k] = _t2n(v).reshape(len(self.request_clients), self.envs_per_group,
                                                            *v.shape[1:])

            with timing.add_time('inference/advance_buffer_indices'):
                if self.buffer.policy_id == self.policy_id and self.phase != PolicyWorkerPhase.STOP:
                    insert_data = {k: v for k, v in policy_outputs.items() if 'rnn_states' not in k}
                    insert_data = {**insert_data, **envstep_outputs}
                    slot_ids, ep_steps, masks, active_masks, valid_choose = self.buffer.advance_indices(timing, self.request_clients, pause=(self.phase == PolicyWorkerPhase.CLOSING), **insert_data)
                    insert_data.pop('dones')

            with timing.add_time('inference/copy_actions'):
                for i, (split_idx, group_idx) in enumerate(organized_requests):
                    if self.buffer.policy_id == self.policy_id and self.phase == PolicyWorkerPhase.CLOSING and not valid_choose[i]:
                        with self.stop_experience_collection_cond:
                            # if in the closing phase current buffer slot is filled, ignore copying action to actor workers,
                            # stop actor experience collection, and prepare to reset actors for evaluation
                            for local_actor_idx in range(self.cfg.actor_group_size):
                                global_actor_idx = self.cfg.actor_group_size * group_idx + local_actor_idx
                                self.stop_experience_collection_cnt[global_actor_idx] += 1
                                if self.stop_experience_collection_cnt[global_actor_idx] >= self.cfg.num_splits and self.replicate_rank == 0:
                                    self.actor_queues[global_actor_idx].put(TaskType.PAUSE)
                            
                            # when all actor workers stop experience collection, ignore all buffer related operations
                            if self.stop_experience_collection_cnt.sum() >= self.cfg.num_splits * self.num_actors:
                                self.stop_experience_collection_cond.notify(1)
                        continue

                    for local_actor_idx in range(self.cfg.actor_group_size):
                        global_actor_idx = self.cfg.actor_group_size * group_idx + local_actor_idx
                        env_slice = slice(local_actor_idx * self.envs_per_split,
                                          (local_actor_idx + 1) * self.envs_per_split)
                        self.act_shms[global_actor_idx][split_idx][:, self.agent_idx] = policy_outputs['actions'][
                            i, env_slice]
                        self.act_semaphores[global_actor_idx][split_idx].release()

            with timing.add_time('inference/insert'):
                if self.buffer.policy_id == self.policy_id and self.phase != PolicyWorkerPhase.STOP:
                    self.buffer.insert(timing, slot_ids, ep_steps, valid_choose, masks=masks, active_masks=active_masks, **insert_data)

                # copy rnn states into small shared memory block
                for k, shm_pairs in self.envstep_output_shms.items():
                    if 'rnn_states' in k:
                        for i, (split_idx, group_idx) in enumerate(organized_requests):
                            shm_pairs[group_idx][split_idx][:] = policy_outputs[k][i]

        self.request_clients = []
        self.total_num_samples += rollout_bs
        self.total_inference_steps += 1

    # noinspection PyProtectedMember
    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        psutil.Process().nice(min(self.cfg.default_niceness + 2, 20))

        # we assume all gpus are available and no gpu is occupies by jobs of other users
        # allocate GPU for all policy workers in a Round-Robin pattern
        cuda_envvars_for_policy(self.worker_idx, 'inference')
        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        self._init(timing)

        last_report = last_cache_cleanup = time.time()
        last_report_samples = 0

        # very conservative limit on the minimum number of requests to wait for
        # this will almost guarantee that the system will continue collecting experience
        # at max rate even when 2/3 of workers are stuck for some reason (e.g. doing a long env reset)
        # Although if your workflow involves very lengthy operations that often freeze workers, it can be beneficial
        # to set min_num_requests to 1 (at a cost of potential inefficiency, i.e. policy worker will use very small
        # batches)
        min_num_requests = self.cfg.min_num_requests
        if min_num_requests is None or min_num_requests == -1:
            min_num_requests = self.num_actor_groups // self.cfg.num_policy_workers
            min_num_requests //= 3
            min_num_requests = max(1, min_num_requests)
        log.info('Min num requests: %d', min_num_requests)

        # Again, very conservative timer. Only wait a little bit, then continue operation.
        wait_for_min_requests = 0.025
        num_requests = deque(maxlen=100)

        while not self.terminate:
            try:
                waiting_started = time.time()
                with timing.time_avg('waiting_avg'), timing.add_time('waiting'):
                    while len(self.request_clients
                              ) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                        try:
                            policy_requests = self.policy_queue.get_many(timeout=0.005)
                            self.request_clients.extend(policy_requests)
                        except Empty:
                            pass

                if len(self.request_clients) > 0:
                    num_requests.append(len(self.request_clients))

                    with timing.add_time('update_weights'):
                        self.maybe_update_weights(timing)

                    with timing.time_avg('inference_avg'), timing.add_time('inference'):
                        self._handle_policy_steps(timing)

                with timing.add_time('get_tasks'):
                    try:
                        task_type = self.task_queue.get_nowait()

                        if task_type == TaskType.TERMINATE:
                            self.terminate = True
                        
                        if task_type == TaskType.PAUSE and self.policy_id == self.buffer.policy_id:
                            self.phase = PolicyWorkerPhase.CLOSING

                        if task_type == TaskType.RESUME:
                            self.phase = PolicyWorkerPhase.WORKING
                        
                        if task_type == TaskType.EVALUATION:
                            # this ensures all policy workers have the same model weights
                            self.maybe_update_weights()
                            self.report_queue.put(dict(policy_id=self.policy_id,
                                replicate_rank=self.replicate_rank,
                                policy_version=self.local_policy_version,))

                            self.phase = PolicyWorkerPhase.STOP

                        self.task_queue.task_done()
                    except Empty:
                        pass

                with timing.add_time('report'):
                    if time.time() - last_report > 3.0 and 'inference_avg' in timing:
                        timing_stats = dict(wait_policy=timing.waiting_avg.value,
                                            step_policy=timing.inference_avg.value)
                        samples_since_last_report = self.total_num_samples - last_report_samples

                        stats = memory_stats('policy_worker', self.device)

                        self.report_queue.put(
                            dict(
                                timing=timing_stats,
                                samples=samples_since_last_report,
                                stats=stats,
                                policy_id=self.policy_id,
                                replicate_rank=self.replicate_rank,
                                policy_version=self.local_policy_version,
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

        time.sleep(0.2)
        log.info('Policy worker avg requests: %.2f, total num sample: %d, total inference steps: %d, timing: %s',
                 np.mean(num_requests), self.total_num_samples, self.total_inference_steps, timing)

    def start_process(self):
        self.process.start()

    def close(self):
        self.task_queue.put(TaskType.TERMINATE)

    def join(self):
        join_or_kill(self.process)
