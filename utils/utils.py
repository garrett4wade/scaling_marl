import numpy as np
import math
import sys
import torch
import queue
import psutil
from contextlib import contextmanager
import multiprocessing as mp
import logging
import os
from collections import OrderedDict

from colorlog import ColoredFormatter
from utils.get_available_gpus import get_gpus_without_triggering_pytorch_cuda_initialization


class TaskType:
    INIT, TERMINATE, RESET, ROLLOUT, EVALUATION, TRAIN, INIT_MODEL, PBT, UPDATE_ENV_STEPS, EMPTY, PAUSE, RESUME = range(
        12)


class SocketState:
    SEND, RECV = range(2)


# logger reference: https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/utils/utils.py
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter("%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
                             datefmt=None,
                             reset=True,
                             log_colors={
                                 'DEBUG': 'cyan',
                                 'INFO': 'white,bold',
                                 'INFOV': 'cyan,bold',
                                 'WARNING': 'yellow',
                                 'ERROR': 'red,bold',
                                 'CRITICAL': 'red,bg_white',
                             },
                             secondary_log_colors={},
                             style='%')
ch.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


class RWLock:
    """ A lock object that allows many simultaneous "read locks", but
    only one "write lock." """
    def __init__(self):
        self._read_ready = mp.Condition(mp.Lock())
        self._readers = 0

    def acquire_read(self):
        """ Acquire a read lock. Blocks only if a thread has
        acquired the write lock. """
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """ Release a read lock. """
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notify(1)
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """ Acquire a write lock. Blocks until there are no
        acquired read or write locks. """
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """ Release a write lock. """
        self._read_ready.release()

    @contextmanager
    def r_locked(self):
        try:
            self.acquire_read()
            yield
        finally:
            self.release_read()

    @contextmanager
    def w_locked(self):
        try:
            self.acquire_write()
            yield
        finally:
            self.release_write()


def drain_semaphore(s):
    while s.acquire(block=False):
        continue

def drain_queue(queue_obj, n_sentinel=0, guard_sentinel=False):
    """Empty a multiprocessing queue object, with options to protect against
    the delay between ``queue.put()`` and ``queue.get()``.  Returns a list of
    the queue contents.
    With ``n_sentinel=0``, simply call ``queue.get(block=False)`` until
    ``queue.Empty`` exception (which can still happen slightly *after* another
    process called ``queue.put()``).
    With ``n_sentinel>1``, call ``queue.get()`` until `n_sentinel` ``None``
    objects have been returned (marking that each ``put()`` process has finished).
    With ``guard_sentinel=True`` (need ``n_sentinel=0``), stops if a ``None``
    is retrieved, and puts it back into the queue, so it can do a blocking
    drain later with ``n_sentinel>1``.
    """
    contents = list()
    if n_sentinel > 0:  # Block until this many None (sentinels) received.
        sentinel_counter = 0
        while True:
            obj = queue_obj.get()
            if obj is None:
                sentinel_counter += 1
                if sentinel_counter >= n_sentinel:
                    return contents
            else:
                contents.append(obj)
    while True:  # Non-blocking, beware of delay between put() and get().
        try:
            obj = queue_obj.get(block=False)
        except queue.Empty:
            return contents
        if guard_sentinel and obj is None:
            # Restore sentinel, intend to do blocking drain later.
            queue_obj.put(None)
            return contents
        elif obj is not None:  # Ignore sentinel.
            contents.append(obj)


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm()**2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def get_obs_shapes_from_spaces(obs_space, share_obs_space):
    obs_shape = get_shape_from_obs_space(obs_space)
    share_obs_shape = get_shape_from_obs_space(share_obs_space)

    if type(obs_shape[-1]) == list:
        obs_shape = obs_shape[:1]

    if type(share_obs_shape[-1]) == list:
        share_obs_shape = share_obs_shape[:1]

    return obs_shape, share_obs_shape


def assert_same_obs_shape(controlled_agents, observation_space, share_observation_space):
    obs_space = observation_space[controlled_agents[0]]
    share_obs_space = share_observation_space[controlled_agents[0]]

    obs_shape, share_obs_shape = get_obs_shapes_from_spaces(obs_space, share_obs_space)

    for agent_id in controlled_agents[1:]:
        cur_obs_shape, cur_share_obs_shape = get_obs_shapes_from_spaces(observation_space[agent_id],
                                                                        share_observation_space[agent_id])

        assert (cur_obs_shape == obs_shape and cur_share_obs_shape
                == share_obs_shape), 'agents controlled by the same policy must have same observation/action shapes!'


def assert_same_act_dim(action_space):
    act_dim = get_shape_from_act_space(action_space[0])

    for cur_act_space in action_space[1:]:
        cur_act_dim = get_shape_from_act_space(cur_act_space)
        assert cur_act_dim == act_dim, 'all agents must have the same action space!'


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def safe_put(q, msg, attempts=3, queue_name=''):
    for attempt in range(attempts):
        try:
            q.put(msg)
            return
        except queue.Full:
            log.warning('Could not put msg to queue, the queue %s is full! Attempt %d', queue_name, attempt)

    log.error('Failed to put msg to queue %s after %d attempts. The message is lost!', queue_name, attempts)


def iterate_recursively(d):
    """
    Generator for a dictionary that can potentially include other dictionaries.
    Yields tuples of (dict, key, value), where key, value are "leaf" elements of the "dict".

    """
    for k, v in d.items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iterate_recursively(v)
        else:
            yield d, k, v


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def join_or_kill(process, timeout=1.0):
    process.join(timeout)
    if process.is_alive():
        log.warning('Process %r could not join, kill it with fire!', process)
        process.kill()
        log.warning('Process %r is dead (%r)', process, process.is_alive())


def list_child_processes():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    is_alive = []
    for child in children:
        try:
            child_process = psutil.Process(child.pid)
            if child_process.is_running():
                is_alive.append(child_process)
        except psutil.NoSuchProcess:
            pass

    return is_alive


def kill_processes(processes):
    for p in processes:
        try:
            if 'torch_shm' in p.name():
                # do not kill to avoid permanent memleaks
                # https://pytorch.org/docs/stable/multiprocessing.html#file-system-file-system
                continue

            # log.debug('Child process name %d %r %r %r', p.pid, p.name(), p.exe(), p.cmdline())
            log.debug('Child process name %d %r %r', p.pid, p.name(), p.exe())
            if p.is_running():
                log.debug('Killing process %s...', p.name())
                p.kill()
        except psutil.NoSuchProcess:
            # log.debug('Process %d is already dead', p.pid)
            pass


def set_global_cuda_envvars(cfg):
    if not cfg.cuda:
        available_gpus = ''
    else:
        available_gpus = get_gpus_without_triggering_pytorch_cuda_initialization(cfg.cwd, os.environ)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = available_gpus
    os.environ['CUDA_VISIBLE_DEVICES_backup_'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def get_available_gpus():
    orig_visible_devices = os.environ['CUDA_VISIBLE_DEVICES_backup_']
    available_gpus = [int(g) for g in orig_visible_devices.split(',') if g]
    return available_gpus


def set_gpus_for_process(process_idx, num_gpus_per_process, process_type, gpu_mask=None):
    available_gpus = get_available_gpus()
    if gpu_mask is not None:
        assert len(available_gpus) >= len(gpu_mask)
        available_gpus = [available_gpus[g] for g in gpu_mask]
    num_gpus = len(available_gpus)
    gpus_to_use = []

    if num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        log.debug('Not using GPUs for %s process %d', process_type, process_idx)
    else:
        first_gpu_idx = process_idx * num_gpus_per_process
        # round-robin gpu allocation
        for i in range(num_gpus_per_process):
            index_mod_num_gpus = (first_gpu_idx + i) % num_gpus
            gpus_to_use.append(available_gpus[index_mod_num_gpus])

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus_to_use])
        log.info(
            'Set environment var %s to %r for %s process %d',
            'CUDA_VISIBLE_DEVICES',
            os.environ['CUDA_VISIBLE_DEVICES'],
            process_type,
            process_idx,
        )
        log.debug('Visible devices: %r', torch.cuda.device_count())

    return gpus_to_use


def cuda_envvars_for_policy(policy_id, process_type):
    set_gpus_for_process(policy_id, 1, process_type)


def memory_consumption_mb():
    """Memory consumption of the current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def memory_stats(process, device):
    memory_mb = memory_consumption_mb()
    stats = {f'memory_{process}': memory_mb}
    if device.type != 'cpu':
        gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
        gpu_cache_mb = torch.cuda.memory_reserved(device) / 1e6
        stats.update({f'gpu_mem_{process}': gpu_mem_mb, f'gpu_cache_{process}': gpu_cache_mb})

    return stats


def cores_for_worker_process(worker_idx, num_workers, cpu_count):
    """
    Returns core indices, assuming available cores are [0, ..., cpu_count).
    If this is not the case (e.g. SLURM) use these as indices in the array of actual available cores.
    """

    worker_idx_modulo = worker_idx % cpu_count

    # trying to optimally assign workers to CPU cores to minimize context switching
    # logic here is best illustrated with an example
    # 20 cores, 44 workers (why? I don't know, someone wanted 44 workers)
    # first 40 are assigned to a single core each, remaining 4 get 5 cores each

    cores = None
    whole_workers_per_core = num_workers // cpu_count
    if worker_idx < whole_workers_per_core * cpu_count:
        # these workers get an private core each
        cores = [worker_idx_modulo]
    else:
        # we're dealing with some number of workers that is less than # of cpu cores
        remaining_workers = num_workers % cpu_count
        if cpu_count % remaining_workers == 0:
            cores_to_use = cpu_count // remaining_workers
            cores = list(range(worker_idx_modulo * cores_to_use, (worker_idx_modulo + 1) * cores_to_use, 1))

    return cores


def set_process_cpu_affinity(worker_idx, num_workers):
    if sys.platform == 'darwin':
        log.debug('On MacOS, not setting affinity')
        return

    curr_process = psutil.Process()
    available_cores = curr_process.cpu_affinity()
    cpu_count = len(available_cores)
    core_indices = cores_for_worker_process(worker_idx, num_workers, cpu_count)
    if core_indices is not None:
        curr_process_cores = [available_cores[c] for c in core_indices]
        curr_process.cpu_affinity(curr_process_cores)

    log.debug('Worker %d uses CPU cores %r', worker_idx, curr_process.cpu_affinity())
