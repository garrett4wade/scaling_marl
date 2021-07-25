import signal
import time
from queue import Empty

import psutil
import torch
import numpy as np
from torch.multiprocessing import Process as TorchProcess

from utils.timing import Timing
from utils.utils import log, memory_consumption_mb, join_or_kill, set_process_cpu_affinity, safe_put, \
    TaskType, set_gpus_for_process

# TODO: may be accelerated by c++ threading pool
from envs.env_wrappers import ShareDummyVecEnv


class ActorWorker:
    """
    Top-level class defining the actor worker (rollout worker in the paper, sorry for the confusion, too lazy to rename)

    Works with an array (vector) of environments that is processes in portions.
    Simple case, env vector is split into two parts:
    1. Do an environment step in the 1st half of the vector (envs 1..N/2)
    2. Send observations to a queue for action generation elsewhere (e.g. on a GPU worker)
    3. Immediately start processing second half of the vector (envs N/2+1..N)
    4. By the time second half is processed, actions for the 1st half should be ready. Immediately start processing
    the 1st half of the vector again.

    As a result, if action generation is fast enough, this env runner should be busy 100% of the time
    calculating env steps, without waiting for actions.
    This is somewhat similar to double-buffered rendering in computer graphics.

    """
    def __init__(
        self,
        cfg,
        env_fn,
        num_agents,
        worker_idx,
        buffer,
        task_queue,
        policy_queue,
        report_queue,
        act_shm,
        act_semaphore,
        envstep_output_shm,
        envstep_output_semaphore,
    ):
        self.cfg = cfg
        self.env_fn = env_fn
        self.num_agents = num_agents

        self.worker_idx = worker_idx

        self.envs_per_actor = cfg.envs_per_actor
        self.num_splits = cfg.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.envs_per_split = self.envs_per_actor // self.num_splits

        self.buffer = buffer
        self.summary_keys = self.buffer.summary_keys
        self.summary_offset = self.worker_idx * self.envs_per_split

        self.initialized = False
        self.terminate = False

        self.group_local_idx = self.worker_idx % self.cfg.actor_group_size
        self.env_slice = slice(self.group_local_idx * self.envs_per_split,
                               (self.group_local_idx + 1) * self.envs_per_split)

        self.env_runners = None

        self.policy_queue = policy_queue
        self.report_queue = report_queue
        self.task_queue = task_queue

        self.act_shm = act_shm
        self.act_semaphore = act_semaphore

        self.envstep_output_shm = envstep_output_shm
        self.envstep_output_semaphore = envstep_output_semaphore

        self.processed_envsteps = 0
        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        """
        Initialize env runners, that actually do all the work. Also we're doing some utility stuff here, e.g.
        setting process affinity (this is a performance optimization).
        """

        log.info('Initializing envs for env runner %d...', self.worker_idx)

        if self.cfg.force_envs_single_thread:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1, user_api=None)

        if self.cfg.set_workers_cpu_affinity:
            set_process_cpu_affinity(self.worker_idx, self.cfg.num_actors)
        psutil.Process().nice(min(self.cfg.default_niceness + 10, 20))

        self.client_ids = [i + self.num_splits * self.worker_idx for i in range(self.num_splits)]
        self.env_runners = []
        for i in range(self.num_splits):
            self.env_runners.append(
                ShareDummyVecEnv([
                    lambda: self.env_fn(self.worker_idx * self.envs_per_actor + i * self.envs_per_split + j, self.cfg)
                    for j in range(self.envs_per_split)
                ]))
            safe_put(self.report_queue, dict(initialized_env=(self.worker_idx, i)), queue_name='report')

        self.initialized = True

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()

        self.terminate = True

    def _handle_reset(self):
        """
        Reset all envs, one split at a time (double-buffering), and send requests to policy workers to get
        actions for the very first env step.
        """
        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset()
            policy_inputs['rewards'] = np.zeros((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs['dones'] = np.zeros((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs['fct_masks'] = np.ones((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            for k, v in self.envstep_output_shm.items():
                if 'rnn_states' not in k:
                    v[split_idx][self.env_slice] = policy_inputs[k]

        for split_idx in range(len(self.env_runners)):
            self.envstep_output_semaphore[split_idx].release()
            if self.cfg.actor_group_size == 1:
                # when #actors == #groups
                self.policy_queue.put(split_idx * self.cfg.num_actors + self.worker_idx)

        log.info('Finished reset for worker %d', self.worker_idx)
        # TODO: figure out what report queue is doing
        safe_put(self.report_queue, dict(finished_reset=self.worker_idx), queue_name='report')

    def _advance_rollouts(self, split_idx, timing):
        """
        Process incoming request from policy worker. Use the data (policy outputs, actions) to advance the simulation
        by one step on the corresponding VectorEnvRunner.

        If we successfully managed to advance the simulation, send requests to policy workers to get actions for the
        next step. If we completed the entire rollout, also send request to the learner!

        :param data: request from the policy worker, containing actions and other policy outputs
        :param timing: profiling stuff
        """
        env = self.env_runners[split_idx]

        with timing.add_time('env_step/simulation'), timing.time_avg('env_step/simulation_avg'):
            envstep_outputs = env.step(self.act_shm[split_idx])

        with timing.add_time('env_step/copy_outputs'):
            infos = envstep_outputs['infos']
            force_terminations = np.array([[[agent_info.get('force_termination', 0)] for agent_info in info]
                                           for info in infos])
            envstep_outputs['fct_masks'] = 1 - force_terminations
            for k, v in self.envstep_output_shm.items():
                if 'rnn_states' not in k:
                    v[split_idx][self.env_slice] = envstep_outputs[k]
            self.envstep_output_semaphore[split_idx].release()

            if self.cfg.actor_group_size == 1:
                # when #actors == #groups
                self.policy_queue.put(split_idx * self.cfg.num_actors + self.worker_idx)

        with timing.add_time('env_step/summary'):
            dones = envstep_outputs['dones']
            for env_id, (done, info) in enumerate(zip(dones, infos)):
                if not np.all(done):
                    continue
                with self.buffer.summary_lock:
                    for i, sum_key in enumerate(self.summary_keys):
                        self.buffer.summary_block[split_idx, self.summary_offset + env_id, i] = info[0][sum_key]

        self.processed_envsteps += 1

    def _run(self):
        """
        Main loop of the actor worker (rollout worker).
        Process tasks (mainly ROLLOUT_STEP) until we get the termination signal, which usually means end of training.
        Currently there is no mechanism to restart dead workers if something bad happens during training. We can only
        retry on the initial reset(). This is definitely something to work on.
        """
        log.info('Initializing vector env runner %d...', self.worker_idx)

        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.cfg.actor_worker_gpus:
            set_gpus_for_process(
                self.worker_idx,
                num_gpus_per_process=1,
                process_type='actor',
                gpu_mask=self.cfg.actor_worker_gpus,
            )

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        cur_split = 0

        last_report = time.time()
        with torch.no_grad():
            while not self.terminate:
                try:
                    if self.initialized:
                        with timing.add_time('waiting'), timing.time_avg('wait_for_inference'):
                            ready = self.act_semaphore[cur_split].acquire(timeout=0.1)

                        with timing.add_time('env_step'), timing.time_avg('one_env_step'):
                            if ready:
                                self._advance_rollouts(cur_split, timing)
                                cur_split = (cur_split + 1) % self.num_splits

                    with timing.add_time('get_tasks'):
                        try:
                            tasks = self.task_queue.get_many(block=False)
                        except Empty:
                            tasks = []

                    for task in tasks:
                        task_type, data = task

                        if task_type == TaskType.INIT:
                            with timing.add_time('init_env'):
                                self._init()
                            continue

                        if task_type == TaskType.TERMINATE:
                            with timing.add_time('close_env'):
                                self._terminate()
                            break

                        # handling actual workload
                        if self.initialized and task_type == TaskType.RESET:
                            with timing.add_time('first_reset'):
                                self._handle_reset()

                    with timing.add_time('report'):
                        if time.time() - last_report > 5.0 and 'one_env_step' in timing:
                            timing_stats = dict(wait_actor=timing.wait_for_inference, step_actor=timing.one_env_step)
                            memory_mb = memory_consumption_mb()
                            stats = dict(memory_actor=memory_mb)
                            safe_put(self.report_queue, dict(timing=timing_stats, stats=stats), queue_name='report')
                            last_report = time.time()

                except RuntimeError as exc:
                    log.warning('Error while processing data w: %d, exception: %s', self.worker_idx, exc)
                    log.warning('Terminate process...')
                    self.terminate = True
                    safe_put(self.report_queue, dict(critical_error=self.worker_idx), queue_name='report')
                except KeyboardInterrupt:
                    self.terminate = True
                except Exception:
                    log.exception('Unknown exception in rollout worker')
                    self.terminate = True

        if self.worker_idx % 16 == 0:
            time.sleep(0.1)
            log.info(
                'Env runner %d, CPU aff. %r, processed env steps %d, timing %s',
                self.worker_idx,
                psutil.Process().cpu_affinity(),
                self.processed_envsteps,
                timing,
            )

    def init(self):
        self.task_queue.put((TaskType.INIT, None))

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
