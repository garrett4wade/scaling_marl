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
        """
        Ctor.

        :param cfg: global config (all CLI params)
        :param obs_space: observation space (spaces) of the environment
        :param action_space: action space(s)
        :param num_agents: number of agents per env (all env should have the same number of agents right now,
        although it should be easy to fix)
        :param worker_idx: index of this worker process
        :param shared_buffers: shared memory data structures initialized in main process (see shared_buffers.py)
        :param task_queue: queue for incoming messages for THIS particular actor worker. See the task types in the loop
        below, but the most common task is ROLLOUT_STEP, which means "here's your actions, advance simulation by
        one step".
        :param policy_queues: FIFO queues associated with all policies participating in training. We send requests
        for policy queue #N to get actions for envs (agents) that are controlled by policy #N.
        :param report_queue: one-way communication with the main process, various stats and whatnot
        :param learner_queues: one-way communication with the learner, sending trajectory buffers for learning
        """

        self.cfg = cfg
        self.env_fn = env_fn
        self.num_agents = num_agents

        self.worker_idx = worker_idx

        self.buffer = buffer

        self.terminate = False

        self.envs_per_actor = cfg.envs_per_actor
        self.num_splits = cfg.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.envs_per_split = self.envs_per_actor // self.num_splits

        num_actors_per_policy_worker = self.cfg.num_actors // self.cfg.num_policy_workers
        self.local_idx = self.worker_idx % num_actors_per_policy_worker
        self.env_slice = slice(self.local_idx * self.envs_per_split, (self.local_idx + 1) * self.envs_per_split)

        self.env_runners = None

        # TODO: policy queue -> policy queues, to support PBT
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
            for k, v in self.envstep_output_shm[split_idx].items():
                if 'rnn_states' not in k:
                    v[self.env_slice] = policy_inputs[k]
            self.envstep_output_semaphore[split_idx].release()

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

        with timing.add_time('env_step/simulation'):
            envstep_outputs = env.step(self.act_shm[split_idx])

        with timing.add_time('env_step/copy_outputs'):
            infos = envstep_outputs['infos']
            force_terminations = np.array([[[agent_info.get('force_termination', 0)] for agent_info in info]
                                           for info in infos])
            envstep_outputs['fct_masks'] = 1 - force_terminations
            for k, v in self.envstep_output_shm[split_idx].items():
                if 'rnn_states' not in k:
                    v[self.env_slice] = envstep_outputs[k]
            self.envstep_output_semaphore[split_idx].release()

        # TODO: deal with episodic summary data

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
                        if task_type == TaskType.RESET:
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

        if self.worker_idx <= 1:
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

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
