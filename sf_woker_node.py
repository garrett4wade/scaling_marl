"""
Algorithm entry poing.
Methods of the APPO class initiate all other components (rollout & policy workers and learners) in the main thread,
and then fork their separate processes.
All data structures that are shared between processes are also created during the construction of APPO.

This class contains the algorithm main loop. All the actual work is done in separate worker processes, so
the only task of the main loop is to collect summaries and stats from the workers and log/save them to disk.

Hyperparameters specific to policy gradient algorithms are defined in this file. See also algorithm.py.

"""

import math
import multiprocessing
import os
import time
from collections import deque
from queue import Empty

import numpy as np
import torch
from torch.multiprocessing import JoinableQueue as TorchJoinableQueue

from system.actor_worker import ActorWorker
from system.policy_worker import PolicyWorker
from utils.buffer import SharedReplayBuffer
from utils.timing import Timing
from utils.utils import iterate_recursively, log, memory_consumption_mb, set_global_cuda_envvars, \
    list_child_processes, kill_processes, AttrDict

if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue


class ExperimentStatus:
    SUCCESS, FAILURE, INTERRUPTED = range(3)


# custom experiments can define functions to this list to do something extra with the raw episode summaries
# coming from the environments
EXTRA_EPISODIC_STATS_PROCESSING = []

# custom experiments or environments can append functions to this list to postprocess some summaries, or aggregate
# summaries, or do whatever else the user wants
EXTRA_PER_POLICY_SUMMARIES = []

torch.multiprocessing.set_sharing_strategy('file_system')


class SFWorkerNode:
    """Async PPO."""
    def __init__(self, cfg, env_fn):

        # we should not use CUDA in the main thread, only on the workers
        set_global_cuda_envvars(cfg)
        self.cfg = cfg

        self.obs_space = cfg.observation_space
        self.share_obs_space = cfg.share_observation_space
        self.action_space = cfg.action_space
        self.num_agents = cfg.num_agents

        self.env_fn = env_fn

        # shared memory allocation
        self.buffer = SharedReplayBuffer(self.cfg, self.num_agents, self.obs_space, self.share_obs_space,
                                         self.action_space)

        self.actor_workers = []
        self.policy_workers = []

        self.report_queue = MpQueue(40 * 1000 * 1000)
        # TODO: policy queue -> policy queues to support PBT
        self.policy_queue = MpQueue()

        self.policy_avg_stats = dict()
        self.policy_lag = dict()

        self.last_timing = dict()
        self.env_steps = dict()
        self.samples_collected = 0
        self.total_env_steps_since_resume = 0

        # currently this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()
        self.last_experiment_summaries = 0

        self.report_interval = 5.0  # sec
        self.log_interval = self.cfg.log_interval  # sec

        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes

        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))
        self.throughput_stats = deque([], maxlen=5)
        self.avg_stats = dict()
        self.stats = dict()  # regular (non-averaged) stats

        # TODO: add summary writer

    def initialize(self):
        pass

    def finalize(self):
        pass

    def create_actor_worker(self, idx, actor_queue):
        return ActorWorker(
            self.cfg,
            self.env_fn,
            self.num_agents,
            idx,
            self.buffer,
            task_queue=actor_queue,
            policy_queue=self.policy_queue,
            report_queue=self.report_queue,
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
            w.init()
            w.request_reset()
            workers[i] = w
            last_env_initialized[i] = time.time()

        total_num_envs = self.cfg.num_actors * self.cfg.env_per_actor
        envs_initialized = [0] * self.cfg.num_actors
        workers_finished = set()

        while len(workers_finished) < len(workers):
            failed_worker = -1

            try:
                report = self.report_queue.get(timeout=1.0)

                if 'initialized_env' in report:
                    worker_idx, split_idx = report['initialized_env']
                    last_env_initialized[worker_idx] = time.time()
                    envs_initialized[worker_idx] += 1

                    log.debug(
                        'Progress for %d workers: %d/%d envs initialized...',
                        len(indices),
                        sum(envs_initialized),
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

                    log.error('Worker %d is stuck or failed (%.3f). Reset!', w.worker_idx, time_passed)
                    log.debug('Status: %r', w.process.is_alive())
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx])
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

        actor_queues = [MpQueue(2 * 1000 * 1000) for _ in range(self.cfg.num_actors)]

        policy_worker_queues = []
        for i in range(self.cfg.num_policy_workers):
            policy_worker_queues.append(TorchJoinableQueue())

        log.info('Initializing learners...')
        policy_lock = multiprocessing.Lock()
        resume_experience_collection_cv = multiprocessing.Condition()

        log.info('Initializing policy workers...')

        for i in range(self.cfg.num_policy_workers):
            policy_worker = PolicyWorker(
                i,
                self.cfg,
                self.obs_space,
                self.share_obs_space,
                self.action_space,
                self.buffer,
                self.policy_queue,
                actor_queues,
                self.report_queue,
                policy_worker_queues[i],
                policy_lock,
                resume_experience_collection_cv,
            )
            self.policy_workers.append(policy_worker)
            policy_worker.start_process()

        log.info('Initializing actors...')

        # We support actor worker initialization in groups, which can be useful for some envs that
        # e.g. crash when too many environments are being initialized in parallel.
        # Currently the limit is not used since it is not required for any envs supported out of the box,
        # so we parallelize initialization as hard as we can.
        # If this is required for your environment, perhaps a better solution would be to use global locks,
        # like FileLock (see doom_gym.py)
        max_parallel_init = int(1e9)  # might be useful to limit this for some envs
        actor_indices = list(range(self.cfg.num_actors))
        for i in range(0, self.cfg.num_actors, max_parallel_init):
            workers = self.init_subset(actor_indices[i:i + max_parallel_init], actor_queues)
            self.actor_workers.extend(workers)

    def finish_initialization(self):
        """Wait until policy workers are fully initialized."""
        for w in self.policy_workers:
            log.debug('Waiting for policy worker %d to finish initialization...', w.worker_idx)
            w.init()
            log.debug('Policy worker %d initialized!', w.worker_idx)

    def process_report(self, report):
        """Process stats from various types of workers."""

        if 'policy_id' in report:
            policy_id = report['policy_id']

            if 'learner_env_steps' in report:
                if policy_id in self.env_steps:
                    delta = report['learner_env_steps'] - self.env_steps[policy_id]
                    self.total_env_steps_since_resume += delta
                self.env_steps[policy_id] = report['learner_env_steps']

            if 'episodic' in report:
                s = report['episodic']
                for _, key, value in iterate_recursively(s):
                    if key not in self.policy_avg_stats:
                        self.policy_avg_stats[key] = deque(maxlen=self.cfg.stats_avg)

                    self.policy_avg_stats[key].append(value)

                    for extra_stat_func in EXTRA_EPISODIC_STATS_PROCESSING:
                        extra_stat_func(policy_id, key, value, self.cfg)

            if 'train' in report:
                self.report_train_summaries(report['train'], policy_id)

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
        """
        Called periodically (every X seconds, see report_interval).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        if len(self.env_steps) == 0:
            return

        now = time.time()
        self.fps_stats.append((now, self.total_env_steps_since_resume))
        if len(self.fps_stats) <= 1:
            return

        fps = []
        for avg_interval in self.avg_stats_intervals:
            past_moment, past_frames = self.fps_stats[max(0, len(self.fps_stats) - 1 - avg_interval)]
            fps.append((self.total_env_steps_since_resume - past_frames) / (now - past_moment))

        sample_throughput = dict()
        self.throughput_stats.append((now, self.samples_collected))
        if len(self.throughput_stats) > 1:
            past_moment, past_samples = self.throughput_stats[0]
            sample_throughput = (self.samples_collected - past_samples) / (now - past_moment)
        else:
            sample_throughput = math.nan

        total_env_steps = sum(self.env_steps.values())
        self.print_stats(fps, sample_throughput, total_env_steps)

        if time.time() - self.last_experiment_summaries > self.log_interval:
            self.report_experiment_summaries(fps[0], sample_throughput)
            self.last_experiment_summaries = time.time()

    def print_stats(self, fps, sample_throughput, total_env_steps):
        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(f'{int(interval * self.report_interval)} sec: {fps_value:.1f}')
        fps_str = f'({", ".join(fps_str)})'

        samples_per_policy = ', '.join([f'{p}: {s:.1f}' for p, s in sample_throughput.items()])

        lag_stats = self.policy_lag[0]
        lag = AttrDict()
        for key in ['min', 'avg', 'max']:
            lag[key] = lag_stats.get(f'version_diff_{key}', -1)
        policy_lag_str = f'min: {lag.min:.1f}, avg: {lag.avg:.1f}, max: {lag.max:.1f}'

        log.debug(
            'Fps is %s. Total num frames: %d. Throughput: %s. Samples: %d. Policy #0 lag: (%s)',
            fps_str,
            total_env_steps,
            samples_per_policy,
            sum(self.samples_collected),
            policy_lag_str,
        )

        if 'reward' in self.policy_avg_stats:
            policy_reward_stats = []
            reward_stats = self.policy_avg_stats['reward']
            if len(reward_stats) > 0:
                policy_reward_stats.append(f'{np.mean(reward_stats):.3f}')
            log.debug('Avg episode reward: %r', policy_reward_stats)

    def report_train_summaries(self, stats, policy_id):
        for key, scalar in stats.items():
            self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id])
            if 'version_diff' in key:
                self.policy_lag[policy_id][key] = scalar

    def report_experiment_summaries(self, fps, sample_throughput):
        memory_mb = memory_consumption_mb()

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            if policy_id == default_policy:
                self.writers[policy_id].add_scalar('0_aux/_fps', fps, env_steps)
                self.writers[policy_id].add_scalar('0_aux/master_process_memory_mb', float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and self.total_train_seconds > 300):
                        self.writers[policy_id].add_scalar(f'stats/{key}', np.mean(value), env_steps)

                for key, value in self.stats.items():
                    self.writers[policy_id].add_scalar(f'stats/{key}', value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                self.writers[policy_id].add_scalar('0_aux/_sample_throughput', sample_throughput[policy_id], env_steps)

            for key, stat in self.policy_avg_stats.items():
                if len(stat[policy_id]) >= stat[policy_id].maxlen or (len(stat[policy_id]) > 10
                                                                      and self.total_train_seconds > 300):
                    stat_value = np.mean(stat[policy_id])
                    writer = self.writers[policy_id]

                    # custom summaries have their own sections in tensorboard
                    if '/' in key:
                        avg_tag = key
                        min_tag = f'{key}_min'
                        max_tag = f'{key}_max'
                    else:
                        avg_tag = f'0_aux/avg_{key}'
                        min_tag = f'0_aux/avg_{key}_min'
                        max_tag = f'0_aux/avg_{key}_max'

                    writer.add_scalar(avg_tag, float(stat_value), env_steps)

                    # for key stats report min/max as well
                    if key in ('reward', 'true_reward', 'len'):
                        writer.add_scalar(min_tag, float(min(stat[policy_id])), env_steps)
                        writer.add_scalar(max_tag, float(max(stat[policy_id])), env_steps)

            for extra_summaries_func in EXTRA_PER_POLICY_SUMMARIES:
                extra_summaries_func(policy_id, self.policy_avg_stats, env_steps, self.writers[policy_id], self.cfg)

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds

        if self.cfg.benchmark:
            end |= self.total_env_steps_since_resume >= int(2e6)
            end |= sum(self.samples_collected) >= int(1e6)

        return end

    def run(self):
        """
        This function contains the main loop of the algorithm, as well as initialization/cleanup code.

        :return: ExperimentStatus (SUCCESS, FAILURE, INTERRUPTED). Useful in testing.
        """

        status = ExperimentStatus.SUCCESS

        self.init_workers()
        self.finish_initialization()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):
            # noinspection PyBroadException
            try:
                while not self._should_end_training():
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

                        self.update_env_steps_actor()
            except Exception:
                log.exception('Exception in driver loop')
                status = ExperimentStatus.FAILURE
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected in driver loop, exiting...')
                status = ExperimentStatus.INTERRUPTED

        all_workers = self.actor_workers + self.policy_workers

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

        fps = self.total_env_steps_since_resume / timing.experience
        log.info('Collected %r, FPS: %.1f', self.env_steps, fps)
        log.info('Timing: %s', timing)

        # if self._should_end_training():
        #     with open(done_filename(self.cfg), 'w') as fobj:
        #         fobj.write(f'{self.env_steps}')

        time.sleep(0.5)
        log.info('Done!')

        return status
