import math
import multiprocessing
import os
import time
from collections import deque
from queue import Empty, Queue

import torch
import zmq
import itertools

from system.actor_worker import ActorWorker
from system.policy_worker import PolicyWorker
from system.actor_group_manager import ActorGroupManager
from utils.buffer import SharedWorkerBuffer
from utils.timing import Timing
from utils.utils import (log, TaskType, list_child_processes, kill_processes, get_obs_shapes_from_spaces,
                         get_shape_from_act_space, assert_same_obs_shape, assert_same_act_dim)

if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue


class ExperimentStatus:
    SUCCESS, FAILURE, INTERRUPTED = range(3)


torch.multiprocessing.set_sharing_strategy('file_system')


class WorkerTask:
    def __init__(self, cfg, task_rank, policy_id, env_fn, local_ps, param_locks, ps_policy_versions, ps_ready_events,
                 task_finish_event):
        self.cfg = cfg
        self.policy_id = policy_id
        self.task_rank = task_rank

        self.controlled_agents = self.cfg.policy2agents[str(self.policy_id)]
        self.num_agents = len(self.controlled_agents)
        self.num_actors = self.cfg.num_actors // self.cfg.num_tasks_per_node

        self.obs_space = cfg.observation_space[self.controlled_agents[0]]
        self.share_obs_space = cfg.share_observation_space[self.controlled_agents[0]]
        self.action_space = cfg.action_space[self.controlled_agents[0]]

        self.env_fn = env_fn

        # local parameter server on this node, which is a list of shared-memory state dicts
        self.local_ps = local_ps
        # multi-reader-single-writer lock
        self.param_locks = param_locks
        # shared-memory array indicating policy versions
        self.ps_policy_versions = ps_policy_versions
        assert self.ps_policy_versions.is_shared()
        # multiprocessing event indicating whether ps is ready
        self.ps_ready_events = ps_ready_events

        # shared memory allocation
        self.buffer = SharedWorkerBuffer(self.cfg, self.policy_id, self.num_agents, self.obs_space,
                                         self.share_obs_space, self.action_space)

        self.actor_workers = []
        self.policy_workers = []

        # faster_fifo queue is initialized using BYTES!
        self.report_queue = MpQueue(40 * 1000 * 1000)  # 40 MB
        self.policy_queues = [MpQueue(40 * 1000) for _ in range(self.cfg.num_policies)]  # 40 KB

        self.num_actor_groups = self.num_actors // self.cfg.actor_group_size

        self.samples_collected = 0

        # currently this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()

        self.report_interval = 5.0  # sec
        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes

        self.throughput_stats = [deque([], maxlen=stat_len) for stat_len in self.avg_stats_intervals]
        self.avg_stats = dict()
        self.stats = dict()  # regular (non-averaged) stats

        self.rollout_policy_ready_events = [[multiprocessing.Event() for _ in range(self.cfg.num_policies)]
                                            for _ in range(self.cfg.num_policy_workers)]

        # shared memory and synchronization primitives for communication between actor worker and policy worker
        self.act_shms, self.act_semaphores = None, None
        self.envstep_output_shms, self.envstep_output_semaphores = None, None

        # ZeroMQ sockets to receive tasks
        self._context = None
        self.task_socket = None
        self.task_result_socket = None

        self.socket_identity = (
            'workertask-' +
            str(self.task_rank + self.cfg.num_tasks_per_node * self.cfg.worker_node_idx)).encode('ascii')
        # TODO: modify this when processing tasks
        self.is_executing_task = False
        self.task_queue = Queue(8)

        self.task_finish_event = task_finish_event

        self.terminate = False

        self.process = multiprocessing.Process(target=self.run)

    def start_process(self):
        self.process.start()

    def init_sockets(self):
        self._context = zmq.Context()

        self.task_socket = self._context.socket(zmq.SUB)
        self.task_socket.connect(self.cfg.task_dispatcher_addr)
        self.task_socket.setsockopt(zmq.SUBSCRIBE, self.socket_identity)

        self.task_result_socket = self._context.socket(zmq.PUSH)
        self.task_result_socket.connect(self.cfg.task_result_addr)
        self.task_result_socket.send(self.socket_identity)

    def init_shm_primitives(self):
        envs_per_split = self.cfg.envs_per_actor // self.cfg.num_splits

        # initialize action/observation shared memories for communication between actors and policy workers
        # TODO: initialize env step outputs using config
        # following is just the case of StarCraft2 (policy-sharing environments)
        envstep_output_keys = [
            'obs', 'share_obs', 'rewards', 'available_actions', 'fct_masks', 'rnn_states', 'rnn_states_critic'
        ]

        # actor workers consume actions and produce envstep_outputs in one shot (in env.step),
        # thus action_shms and envstep_output_semaphores only have one copy (not seperated for different policy_ids)
        self.envstep_output_semaphores = [[multiprocessing.Semaphore(0) for _ in range(self.cfg.num_splits)]
                                          for _ in range(self.num_actors)]

        assert_same_act_dim(self.cfg.action_space)
        act_dim = get_shape_from_act_space(self.cfg.action_space[0])
        self.act_shms = [[
            torch.zeros((envs_per_split, self.cfg.num_agents, act_dim), dtype=torch.float32).share_memory_().numpy()
            for _ in range(self.cfg.num_splits)
        ] for _ in range(self.num_actors)]

        # in the opposite side, different policy workers with different policy ids consume different subsets of
        # envstep_outputs and produce different subsets of actions asynchronously, thus act_semaphores and
        # envstep_output_shms have different copies for different policy ids
        self.act_semaphores = []
        self.envstep_output_shms = {}
        for k in envstep_output_keys:
            self.envstep_output_shms[k] = []
        self.envstep_output_shms['dones'] = []

        for controlled_agents in self.cfg.policy2agents.values():
            self.act_semaphores.append([[multiprocessing.Semaphore(0) for _ in range(self.cfg.num_splits)]
                                        for _ in range(self.num_actors)])

            assert_same_obs_shape(controlled_agents, self.cfg.observation_space, self.cfg.share_observation_space)
            obs_shape, share_obs_shape = get_obs_shapes_from_spaces(
                self.cfg.observation_space[controlled_agents[0]],
                self.cfg.share_observation_space[controlled_agents[0]])
            num_agents = len(controlled_agents)

            for k in envstep_output_keys:
                if not hasattr(self.buffer, k):
                    continue

                if k == 'obs':
                    shape = obs_shape
                elif k == 'share_obs':
                    shape = share_obs_shape
                else:
                    shape = getattr(self.buffer, k).shape[4:]
                shape = (envs_per_split * self.cfg.actor_group_size, num_agents, *shape)

                self.envstep_output_shms[k].append([[
                    torch.zeros(shape, dtype=torch.float32).share_memory_().numpy() for _ in range(self.cfg.num_splits)
                ] for _ in range(self.num_actor_groups)])

            dones_shape = (envs_per_split * self.cfg.actor_group_size, num_agents, 1)
            self.envstep_output_shms['dones'].append([[
                torch.zeros(dones_shape, dtype=torch.float32).share_memory_().numpy()
                for _ in range(self.cfg.num_splits)
            ] for _ in range(self.num_actor_groups)])

    def create_actor_worker(self, idx, actor_queue):
        group_idx = idx // self.cfg.actor_group_size
        return ActorWorker(
            self.cfg,
            self.task_rank,
            idx,
            self.env_fn,
            self.buffer,
            actor_queue,
            self.policy_queues,
            self.report_queue,
            self.act_shms[idx],
            [policy_act_semaphores[idx] for policy_act_semaphores in self.act_semaphores],
            {k: [policy_v[group_idx] for policy_v in v]
             for k, v in self.envstep_output_shms.items()},
            self.envstep_output_semaphores[idx],
        )

    # noinspection PyProtectedMember
    def init_subset(self, indices, actor_queues):
        """
        Initialize a subset of actor workers (rollout workers) and wait until the first reset() is completed for all
        envs on these workers.

        This function will retry if the worker process crashes during the initial reset.

        :param indices: indices of actor workers to initialize
        :param actor_queues: task queues corresponding to these workers
        :return: initialized workers
        """

        reset_timelimit_seconds = self.cfg.reset_timeout_seconds

        workers = dict()
        last_env_initialized = dict()
        for i in indices:
            w = self.create_actor_worker(i, actor_queues[i])
            w.start_process()
            w.init()
            w.request_reset()
            workers[i] = w
            last_env_initialized[i] = time.time()

        total_num_envs = self.num_actors * self.cfg.envs_per_actor
        envs_initialized = [0] * self.num_actors
        workers_finished = set()

        while len(workers_finished) < len(workers):
            failed_worker = -1

            try:
                report = self.report_queue.get(timeout=1.0)

                if 'initialized_env' in report:
                    worker_idx, split_idx = report['initialized_env']
                    last_env_initialized[worker_idx] = time.time()
                    envs_initialized[worker_idx] += self.cfg.envs_per_actor // self.cfg.num_splits

                    log.debug(
                        'Progress for %d workers: %d/%d envs initialized...',
                        len(indices),
                        sum(envs_initialized) + indices[0] * self.cfg.envs_per_actor,
                        total_num_envs,
                    )
                elif 'finished_reset' in report:
                    workers_finished.add(report['finished_reset'])
                elif 'critical_error' in report:
                    failed_worker = report['critical_error']
            except Empty:
                pass

            for worker_idx, w in workers.items():
                if worker_idx in workers_finished:
                    continue

                time_passed = time.time() - last_env_initialized[worker_idx]
                timeout = time_passed > reset_timelimit_seconds

                if timeout or failed_worker == worker_idx or not w.process.is_alive():
                    envs_initialized[worker_idx] = 0

                    log.error('Worker %d is stuck or failed (%.3f). Reset!', w.local_rank, time_passed)
                    log.debug('Status: %r', w.process.is_alive())
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx])
                    new_worker.start_process()
                    new_worker.init()
                    new_worker.request_reset()

                    last_env_initialized[worker_idx] = time.time()
                    workers[worker_idx] = new_worker
                    del stuck_worker

        return workers.values()

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        """
        Initialize all types of workers and start their worker processes.
        """

        actor_queues = [MpQueue(2 * 1000 * 1000) for _ in range(self.num_actors)]

        if self.cfg.actor_group_size > 1:
            log.info('Initializing group managers...')
            self.group_managers = []
            for idx in range(self.num_actor_groups):
                s = slice(idx * self.cfg.actor_group_size, (idx + 1) * self.cfg.actor_group_size)
                gm = ActorGroupManager(self.cfg, idx, self.policy_queues, self.envstep_output_semaphores[s])
                gm.start()
                self.group_managers.append(gm)

        log.info('Initializing policy workers...')
        for i in range(self.cfg.num_policy_workers):
            policy_worker_tuple = []
            for policy_id in range(self.cfg.num_policies):
                policy_worker = PolicyWorker(
                    self.cfg,
                    policy_id,
                    self.task_rank,
                    i,
                    self.cfg.observation_space[policy_id],
                    self.cfg.share_observation_space[policy_id],
                    self.cfg.action_space[policy_id],
                    self.buffer,
                    self.policy_queues[policy_id],
                    self.report_queue,
                    self.act_shms,
                    self.act_semaphores[policy_id],
                    {k: v[policy_id]
                     for k, v in self.envstep_output_shms.items()},
                    self.rollout_policy_ready_events[i][policy_id],
                    self.local_ps[policy_id],
                    self.param_locks[policy_id],
                    self.ps_policy_versions[policy_id:policy_id + 1],
                    self.ps_ready_events[policy_id],
                )
                policy_worker_tuple.append(policy_worker)
                policy_worker.start_process()
            self.policy_workers.append(tuple(policy_worker_tuple))

        log.info('Initializing actors...')

        # We support actor worker initialization in groups, which can be useful for some envs that
        # e.g. crash when too many environments are being initialized in parallel.
        # Currently the limit is not used since it is not required for any envs supported out of the box,
        # so we parallelize initialization as hard as we can.
        # If this is required for your environment, perhaps a better solution would be to use global locks,
        # like FileLock (see doom_gym.py)
        max_parallel_init = int(1e9)  # might be useful to limit this for some envs
        actor_indices = list(range(self.num_actors))
        for i in range(0, self.num_actors, max_parallel_init):
            workers = self.init_subset(actor_indices[i:i + max_parallel_init], actor_queues)
            self.actor_workers.extend(workers)

    def finish_initialization(self):
        """Wait until policy workers are fully initialized."""
        for w in list(itertools.chain(*self.policy_workers)):
            log.debug('Waiting for policy worker %d to finish initialization...', w.worker_idx)
            w.initialized_event.wait()
            log.debug('Policy worker %d initialized!', w.worker_idx)

    def process_report(self, report):
        """Process stats from various types of workers."""

        if 'samples' in report:
            self.samples_collected += report['samples']

        if 'timing' in report:
            for k, v in report['timing'].items():
                if k not in self.avg_stats:
                    self.avg_stats[k] = deque([], maxlen=50)
                self.avg_stats[k].append(v)

        if 'stats' in report:
            self.stats.update(report['stats'])

    def report(self):
        # TODO: send summary info to task_result_socket during evaluation
        """
        Called periodically (every X seconds, see report_interval).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """
        now = time.time()
        sample_throughputs = []
        for throughput_stat in self.throughput_stats:
            throughput_stat.append((now, self.samples_collected))
            if len(throughput_stat) > 1:
                past_moment, past_samples = throughput_stat[0]
                sample_throughput = (self.samples_collected - past_samples) / (now - past_moment)
            else:
                sample_throughput = math.nan
            sample_throughputs.append(sample_throughput)

        self.print_stats(sample_throughputs)

    def print_stats(self, sample_throughputs):
        log.debug('Throughput: {:.2f} (10 sec), {:.2f} (1 min), {:.2f} (5 mins). Samples: {}.'.format(
            *sample_throughputs, self.samples_collected))
        mem_stats = ''.join(['{}: {:.2f} MB, '.format(k, v) for k, v in self.stats.items()])[:-2]
        log.debug('Memory statistics: %s', mem_stats)
        timing_stats = ''.join(['{}: {:.4f} s, '.format(k, sum(v) / len(v)) for k, v in self.avg_stats.items()])[:-2]
        log.debug('Timing: %s', timing_stats)

    def process_task(self, task):
        # TODO: modify self.is_executing_tasks when processing tasks
        if task == TaskType.ROLLOUT:
            pass
        elif task == TaskType.TERMINATE:
            self.terminate = True
        else:
            raise NotImplementedError

    def run(self):
        """
        This function contains the main loop of the algorithm, as well as initialization/cleanup code.

        :return: ExperimentStatus (SUCCESS, FAILURE, INTERRUPTED). Useful in testing.
        """

        status = ExperimentStatus.SUCCESS

        self.init_shm_primitives()
        self.init_workers()

        self.finish_initialization()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):
            try:
                while not self.terminate:
                    try:
                        msg = self.task_socket.recv_multipart(flags=zmq.NOBLOCK)
                        self.task_queue.put(msg)
                    except zmq.ZMQError:
                        pass

                    if not self.is_executing_task:
                        try:
                            # TODO: here we don't process task except for TERMINATE
                            msg = self.task_queue.get()
                            task = int(msg[1].decode('ascii'))
                            self.process_task(task)
                            if self.terminate:
                                break
                        except Empty:
                            log.warning('Trainer %d is not executing tasks and there are no tasks distributed to it!',
                                        self.trainer_idx)
                            pass

                    try:
                        reports = self.report_queue.get_many(timeout=0.1)
                        for report in reports:
                            self.process_report(report)
                    except Empty:
                        pass

                    if time.time() - self.last_report > self.report_interval:
                        self.report()

                        now = time.time()
                        self.total_train_seconds += now - self.last_report
                        self.last_report = now

            except Exception:
                log.exception('Exception in driver loop')
                status = ExperimentStatus.FAILURE
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected in driver loop, exiting...')
                status = ExperimentStatus.INTERRUPTED

        all_workers = self.actor_workers + list(itertools.chain(*self.policy_workers))

        child_processes = list_child_processes()

        time.sleep(0.1)
        log.debug('Closing workers...')
        for i, w in enumerate(all_workers):
            w.close()
            time.sleep(0.01)
        for i, w in enumerate(all_workers):
            w.join()
        log.debug('Workers joined!')

        # VizDoom processes often refuse to die for an unidentified reason, so we're force killing them with a hack
        kill_processes(child_processes)

        fps = self.samples_collected / timing.experience
        log.info('Collected %r, FPS: %.1f', self.samples_collected, fps)
        log.info('Timing: %s', timing)

        time.sleep(0.5)
        log.info('Done!')

        self.task_finish_event.set()

        return status
