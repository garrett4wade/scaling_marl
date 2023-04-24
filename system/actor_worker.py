import signal
import time
from queue import Empty

import psutil
import torch
import numpy as np
from torch.multiprocessing import Process as TorchProcess

from utils.timing import Timing
from utils.utils import log, memory_consumption_mb, join_or_kill, set_process_cpu_affinity, safe_put, \
    TaskType, set_gpus_for_process, drain_semaphore

from envs.env_wrappers import DummyVecEnv
import zmq


def flatten_recurrent(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            tmp = flatten_recurrent(v)
            for k in result.keys():
                assert k not in tmp, k
            result = {**result, **tmp}
        else:
            assert k not in result.keys(), k
            result[k] = v
    return result


class ActorWorkerPhase:
    ROLLOUT, EVALUATION, PAUSE = range(3)


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
        task_rank,
        local_rank,
        env_fn,
        buffer,
        task_queue,
        policy_queues,
        report_queue,
        act_shm,
        act_semaphore,
        envstep_output_shm,
        envstep_output_semaphore,
        eval_summary_block,
        eval_summary_lock,
        eval_episode_cnt,
        eval_finish_event,
        pause_alignment,
    ):
        self.cfg = cfg
        self.env_fn = env_fn
        self.policy2agents = self.cfg.policy2agents
        self.agent_ids = [controlled_agents for controlled_agents in self.policy2agents.values()]
        self.agent_numbers = [len(controlled_agents) for controlled_agents in self.policy2agents.values()]
        self.num_agents = sum(self.agent_numbers)

        self.task_rank = task_rank
        self.local_rank = local_rank
        self.num_actors = self.cfg.num_actors // self.cfg.num_tasks_per_node
        self.worker_idx = self.num_actors * self.task_rank + self.local_rank

        self.envs_per_actor = cfg.envs_per_actor
        self.num_splits = cfg.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.envs_per_split = self.envs_per_actor // self.num_splits

        self.buffer = buffer
        self.env_summary_keys = self.buffer.env_summary_keys
        self.summary_offset = self.local_rank * self.envs_per_split

        self.initialized = False
        self.terminate = False
        self.ready = False

        self.group_local_idx = self.local_rank % self.cfg.actor_group_size
        self.env_slice = slice(self.group_local_idx * self.envs_per_split,
                               (self.group_local_idx + 1) * self.envs_per_split)

        self.env_runners = None

        self.policy_queues = policy_queues
        self.report_queue = report_queue
        self.task_queue = task_queue

        self.act_shm = act_shm
        self.act_semaphore = act_semaphore
        self.is_policy_act_semaphores_ready = np.zeros(self.cfg.num_policies, dtype=np.bool)

        self.envstep_output_shm = envstep_output_shm
        self.envstep_output_semaphore = envstep_output_semaphore

        self.phase = ActorWorkerPhase.PAUSE
        self.summary_phase = ActorWorkerPhase.ROLLOUT

        self.eval_summary_block = eval_summary_block
        self.eval_summary_lock = eval_summary_lock
        self.eval_episode_cnt = eval_episode_cnt
        self.eval_finish_event = eval_finish_event

        self.pause_alignment = pause_alignment
        # just for debugging in some assertion
        self.debug_ep_steps = np.zeros(self.num_splits)

        self.processed_envsteps = 0
        self.process = TorchProcess(target=self._run, daemon=True)

        # cl
        self.reset_tasks_queue = []

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

        self.env_runners = []
        for i in range(self.num_splits):
            self.env_runners.append(
                DummyVecEnv([
                    lambda: self.env_fn(self.worker_idx * self.envs_per_actor + i * self.envs_per_split + j, self.worker_idx * self.envs_per_actor + i * self.envs_per_split, self.cfg)
                    for j in range(self.envs_per_split)
                ]))
            safe_put(self.report_queue, dict(initialized_env=(self.local_rank, i)), queue_name='report')

        # init reset socket, cl
        self._context = zmq.Context()
        socket = self._context.socket(zmq.SUB)
        socket.connect(self.cfg.reset_addrs[0])
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.reset_sockets = [socket]

        self.initialized = True

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()

        self.terminate = True

    # TODO, collect tasks
    def _collect_reset_tasks(self, timing, block=False):
        socket = self.reset_sockets[0]

        with timing.add_time('update_reset_tasks'), timing.time_avg('update_reset_tasks_once'):
            msg = None

            if block:
                msg = socket.recv_multipart(flags=0)
            else:
                while True:
                    # receive the latest reset tasks
                    try:
                        msg = socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
            if msg is not None:
                new_msg = []
                for msg_one in msg:
                    msg_str = msg_one.decode('ascii')
                    if msg_str == 'None':
                        data = None
                    else:
                        data = np.array(msg_str[1:-1].split(),dtype=float)
                    new_msg.append(data)
                self.reset_tasks_queue = new_msg.copy()
    
    def _get_new_reset_tasks(self):
        if len(self.reset_tasks_queue) > 0:
            idx = np.random.randint(0,len(self.reset_tasks_queue), self.num_actors * self.envs_per_actor)
            return np.array(self.reset_tasks_queue)[idx]
        else:
            return None

    # TODO cl, reset task to every env
    def _handle_reset_cl(self, timing):
        """
        Reset all envs, one split at a time (double-buffering), and send requests to policy workers to get
        actions for the very first env step.
        """
        for s in self.envstep_output_semaphore:
            drain_semaphore(s)

        for s_pair in self.act_semaphore:
            for s in s_pair:
                drain_semaphore(s)

        self.is_policy_act_semaphores_ready[:] = False

        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset()
            policy_inputs['rewards'] = np.zeros((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs['dones'] = np.zeros((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs['fct_masks'] = np.ones((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs = flatten_recurrent(policy_inputs)
            for k, shms in self.envstep_output_shm.items():
                # new rnn states is updated after each inference step
                if 'rnn_states' not in k:
                    for policy_shm, agent_idx in zip(shms, self.agent_ids):
                        policy_shm[split_idx][self.env_slice] = policy_inputs[k][:, agent_idx]

        for split_idx in range(len(self.env_runners)):
            self.envstep_output_semaphore[split_idx].release()
            if self.cfg.actor_group_size == 1:
                # when #actors == #groups
                self.envstep_output_semaphore[split_idx].acquire()
                # assert not self.envstep_output_semaphore[split_idx].acquire(block=False)
                for policy_queue in self.policy_queues:
                    policy_queue.put(split_idx * self.num_actors + self.local_rank)

        if not self.ready:
            log.info('Worker task %d finished reset of worker %d (after initialization)', self.task_rank,
                     self.worker_idx)
            safe_put(self.report_queue, dict(finished_reset=self.local_rank), queue_name='report')
            self.ready = True
        # else:
        #     log.info('Worker task %d finished reset of worker %d (for evaluation)', self.task_rank, self.worker_idx)

    def _handle_reset(self):
        """
        Reset all envs, one split at a time (double-buffering), and send requests to policy workers to get
        actions for the very first env step.
        """
        for s in self.envstep_output_semaphore:
            drain_semaphore(s)

        for s_pair in self.act_semaphore:
            for s in s_pair:
                drain_semaphore(s)

        self.is_policy_act_semaphores_ready[:] = False

        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset()
            policy_inputs['rewards'] = np.zeros((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs['dones'] = np.zeros((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs['fct_masks'] = np.ones((self.envs_per_split, self.num_agents, 1), dtype=np.float32)
            policy_inputs = flatten_recurrent(policy_inputs)
            for k, shms in self.envstep_output_shm.items():
                # new rnn states is updated after each inference step
                if 'rnn_states' not in k:
                    for policy_shm, agent_idx in zip(shms, self.agent_ids):
                        policy_shm[split_idx][self.env_slice] = policy_inputs[k][:, agent_idx]

        for split_idx in range(len(self.env_runners)):
            self.envstep_output_semaphore[split_idx].release()
            if self.cfg.actor_group_size == 1:
                # when #actors == #groups
                self.envstep_output_semaphore[split_idx].acquire()
                # assert not self.envstep_output_semaphore[split_idx].acquire(block=False)
                for policy_queue in self.policy_queues:
                    policy_queue.put(split_idx * self.num_actors + self.local_rank)

        if not self.ready:
            log.info('Worker task %d finished reset of worker %d (after initialization)', self.task_rank,
                     self.worker_idx)
            safe_put(self.report_queue, dict(finished_reset=self.local_rank), queue_name='report')
            self.ready = True
        # else:
        #     log.info('Worker task %d finished reset of worker %d (for evaluation)', self.task_rank, self.worker_idx)

    def _advance_rollouts(self, split_idx, timing, tasks=None):
        """
        Process incoming request from policy worker. Use the data (policy outputs, actions) to advance the simulation
        by one step on the corresponding VectorEnvRunner.

        If we successfully managed to advance the simulation, send requests to policy workers to get actions for the
        next step. If we completed the entire rollout, also send request to the learner!

        :param data: request from the policy worker, containing actions and other policy outputs
        :param timing: profiling stuff
        """
        env = self.env_runners[split_idx]
        
        # tasks : self.num_actors * self.envs_per_actor
        if tasks is None:
            env_set_tasks = None
        else:
            # print('tasks',len(tasks), 'local_rank', self.local_rank, 'envs_per_actor', self.envs_per_actor)
            task_chunk = tasks[self.local_rank * self.envs_per_actor : (self.local_rank + 1) * self.envs_per_actor]
            if split_idx == 0:
                env_set_tasks = task_chunk[:self.envs_per_split].copy()
            else :
                env_set_tasks = task_chunk[self.envs_per_split:].copy()

        with timing.add_time('env_step/simulation'), timing.time_avg('env_step/simulation_avg'):
            envstep_outputs = flatten_recurrent(env.step(self.act_shm[split_idx], env_set_tasks))
            self.debug_ep_steps[split_idx] += 1

        with timing.add_time('env_step/copy_outputs'):
            infos = envstep_outputs['infos']
            force_terminations = np.array([[[info.get('force_termination', 0)] for _ in range(self.num_agents)]
                                           for info in infos])
            envstep_outputs['fct_masks'] = 1 - force_terminations
            for k, shms in self.envstep_output_shm.items():
                if 'rnn_states' not in k:
                    for policy_shm, agent_idx in zip(shms, self.agent_ids):
                        policy_shm[split_idx][self.env_slice] = envstep_outputs[k][:, agent_idx]
            self.envstep_output_semaphore[split_idx].release()

            if self.cfg.actor_group_size == 1:
                # when #actors == #groups
                self.envstep_output_semaphore[split_idx].acquire()
                # assert not self.envstep_output_semaphore[split_idx].acquire(block=False)
                for policy_queue in self.policy_queues:
                    policy_queue.put(split_idx * self.num_actors + self.local_rank)

        with timing.add_time('env_step/summary'):
            dones = envstep_outputs['dones'].copy()
            for env_id, (done, info) in enumerate(zip(dones, infos)):
                if not np.all(done):
                    continue
                if self.summary_phase == ActorWorkerPhase.ROLLOUT:
                    with self.buffer.env_summary_lock:
                        for i, sum_key in enumerate(self.env_summary_keys):
                            self.buffer.summary_block[split_idx, self.summary_offset + env_id, i] = info[sum_key]
                elif self.summary_phase == ActorWorkerPhase.EVALUATION:
                    with self.eval_summary_lock:
                        if not self.eval_finish_event.is_set():
                            self.eval_episode_cnt += 1
                            for i, sum_key in enumerate(self.env_summary_keys):
                                self.eval_summary_block[split_idx, self.summary_offset + env_id, i] = info[sum_key]
                            if self.eval_episode_cnt >= self.cfg.eval_episodes:
                                self.eval_finish_event.set()
                else:
                    raise NotImplementedError

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

        cur_split = stop_split = 0

        last_report = time.time()
        with torch.no_grad():
            while not self.terminate:
                try:                    
                    if self.initialized and not (self.phase == ActorWorkerPhase.PAUSE and cur_split == stop_split):
                        with timing.add_time('waiting'), timing.time_avg('wait_for_inference'):
                            for i, policy_act_semaphore in enumerate(self.act_semaphore):
                                if not self.is_policy_act_semaphores_ready[i]:
                                    cur_ready = policy_act_semaphore[cur_split].acquire(timeout=0.05)
                                    # assert not policy_act_semaphore[cur_split].acquire(block=False)
                                    self.is_policy_act_semaphores_ready[i] = cur_ready

                        with timing.add_time('env_step'):
                            # reset env by cl
                            # cl, receive reset tasks
                            self._collect_reset_tasks(timing)
                            # maintain an archive with num_actor * envs_per_actor
                            reset_tasks = self._get_new_reset_tasks()
                            if reset_tasks is None:
                                print('reset_tasks', reset_tasks)
                            else:
                                print('reset_tasks', reset_tasks.shape)
                            # reset_tasks = None
                            if np.all(self.is_policy_act_semaphores_ready):
                                self._advance_rollouts(cur_split, timing, reset_tasks)
                                cur_split = (cur_split + 1) % self.num_splits
                                self.is_policy_act_semaphores_ready[:] = 0

                    if self.phase == ActorWorkerPhase.PAUSE and cur_split == stop_split:
                        assert self.debug_ep_steps[0] == self.debug_ep_steps[1]
                        self.pause_alignment.set()

                    with timing.add_time('get_tasks'):
                        try:
                            tasks = self.task_queue.get_many(block=False)
                        except Empty:
                            tasks = []

                    for task_type in tasks:
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
                            assert self.initialized
                            with timing.add_time('reset'):
                                self._handle_reset()
                                # self._handle_reset_cl(timing)

                        if task_type == TaskType.START:
                            self.phase = self.summary_phase = ActorWorkerPhase.ROLLOUT

                        if task_type == TaskType.EVALUATION:
                            self.phase = self.summary_phase = ActorWorkerPhase.EVALUATION
                            with timing.add_time('evaluation_reset'):
                                self._handle_reset()

                        if task_type == TaskType.PAUSE:
                            assert self.initialized
                            if self.phase == ActorWorkerPhase.EVALUATION:
                                # after evaluation, we need to reset stop_split and debug_steps to
                                # align steps of different environment splits
                                self.debug_ep_steps[:] = 0
                                stop_split = cur_split
                            self.phase = ActorWorkerPhase.PAUSE

                    with timing.add_time('report'):
                        if time.time() - last_report > 5.0 and 'env_step/simulation_avg' in timing:
                            timing_stats = dict(wait_actor=timing.wait_for_inference.value,
                                                step_actor=getattr(timing, 'env_step/simulation_avg').value)
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
        self.task_queue.put(TaskType.INIT)

    def request_reset(self):
        self.task_queue.put(TaskType.RESET)

    def request_eval(self):
        self.task_queue.put(TaskType.EVALUATION)

    def pause(self):
        self.task_queue.put(TaskType.PAUSE)

    def close(self):
        self.task_queue.put(TaskType.TERMINATE)

    def start_rollout(self):
        self.task_queue.put(TaskType.START)

    def join(self):
        join_or_kill(self.process)
