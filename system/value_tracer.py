import psutil
from utils.timing import Timing
import time

from queue import Empty
from utils.utils import log, TaskType
from numpy import float32
import numpy as np

import torch
import torch.multiprocessing as mp

from algorithms.utils.value_head import ValueHead
from algorithms.utils.modules import compute_gae, masked_adv_normalization


class ValueTracer:
    ''' compute value target for trainer use, e.g. n-step bootstrap, GAE, V-trace, Retrace'''
    def __init__(
        self,
        cfg,
        trainer_idx,
        replicate_rank,
        buffer,
        value_tracer_queue,
        batch_queue,
        task_queue,
        shm_state_dict,
        trainer_policy_version,
        param_lock,
    ):
        self.cfg = cfg
        self.num_critic = self.cfg.num_critic
        self.trainer_idx = trainer_idx
        self.replicate_rank = replicate_rank

        self.buffer = buffer
        self.policy_id = buffer.policy_id
        self.value_tracer_idx = self.trainer_idx * cfg.num_value_tracers_per_trainer + self.replicate_rank

        # to receive reanalyzed data
        self.value_tracer_queue = value_tracer_queue
        # to send data to trainer after computing trace
        self.batch_queue = batch_queue

        self.gamma = float32(self.cfg.gamma)
        self.lmbda = float32(self.cfg.gae_lambda)

        # to synchronize PopArt value head weights with trainers
        self.shm_state_dict = shm_state_dict
        self.trainer_policy_version = trainer_policy_version
        self.local_policy_version = -1
        assert self.trainer_policy_version.is_shared()
        for k, v in self.shm_state_dict.items():
            assert v.is_shared()
        self.param_lock = param_lock

        self.update_cnt = 0

        self.task_queue = task_queue

        self.initialized = self.terminate = False

        self.process = mp.Process(target=self._run)

    def start_process(self):
        self.process.start()

    def _init(self, timing):
        # policy network
        self.value_normalizer = ValueHead(self.cfg.hidden_size, 1, self.cfg.use_orthogonal, self.cfg.use_popart)
        self.value_normalizer.eval()

        self.state_dict_keys = self.value_normalizer.state_dict().keys()

        log.debug('Value Tracer {} of trainer {} waiting for all nodes ready...'.format(
            self.replicate_rank, self.trainer_idx))

        with self.param_lock.r_locked():
            with timing.time_avg('update_weights/load_state_dict_once'):
                subset = {k.split('.')[-1]: v for k, v in self.shm_state_dict.items() if k.split('.')[-1] in self.state_dict_keys}
                self.value_normalizer.load_state_dict(subset)
                self.local_policy_version = self.trainer_policy_version.item()

            log.info('Value Tracer %d of trainer %d --- Update to policy version %d', self.replicate_rank,
                    self.trainer_idx, self.local_policy_version)

        self.initialized = True

    def maybe_update_weights(self, timing):
        with self.param_lock.r_locked():
            if self.local_policy_version + self.cfg.sample_reuse * self.cfg.broadcast_interval <= self.trainer_policy_version:
                with timing.time_avg('update_weights/load_state_dict_once'):
                    subset = {k.split('.')[-1]: v for k, v in self.shm_state_dict.items() if k.split('.')[-1] in self.state_dict_keys}
                    self.value_normalizer.load_state_dict(subset)
                    self.local_policy_version = self.trainer_policy_version.item()

                self.update_cnt += 1
                if self.update_cnt % 10 == 0:
                    log.info('Value Tracer %d of trainer %d --- Update to policy version %d', self.replicate_rank,
                            self.trainer_idx, self.local_policy_version)

    @torch.no_grad()
    def trace_step(self, slot_id, timing):
        with self.buffer.trace_lock:
            if self.buffer._is_trace_ready[slot_id]:
                return

        if self.cfg.use_popart:
            self.maybe_update_weights(timing)
            denormalized_values = []
            for critic_id in range(self.num_critic):
                denormalized_values_one = self.value_normalizer.denormalize(np.expand_dims(self.buffer.values[slot_id,:,:,critic_id],axis=-1))
                denormalized_values.append(denormalized_values_one)
            if self.num_critic == 1:
                denormalized_values = np.stack(denormalized_values, axis=-1).squeeze(-1)
            else:
                denormalized_values = np.stack(denormalized_values, axis=-1).squeeze()
        else:
            denormalized_values = self.buffer.values[slot_id]

        # compute value targets and advantages every time learner fetches data, because as the same slot
        # is reused, we need to recompute values of corresponding observations
        with timing.add_time('compute_gae'):
            compute_gae(self.buffer, slot_id, denormalized_values, self.gamma, self.lmbda)

        with timing.add_time('adv_normalization'):
            masked_adv_normalization(self.buffer, slot_id)

        with self.buffer.trace_lock:
            self.buffer._is_trace_ready[slot_id] = 1

    def _run(self):
        psutil.Process().nice(self.cfg.default_niceness + 1)

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        self._init(timing)

        try:
            while not self.terminate:

                with timing.add_time('wait_for_batch'):
                    try:
                        slot_id = self.value_tracer_queue.get(timeout=0.5)
                    except Empty:
                        slot_id = None

                if slot_id is not None:
                    self.trace_step(slot_id, timing)

                    self.batch_queue.put(slot_id)

                with timing.add_time('get_tasks'):
                    try:
                        task_type = self.task_queue.get_nowait()

                        if task_type == TaskType.TERMINATE:
                            self.terminate = True

                        self.task_queue.task_done()
                    except Empty:
                        pass

        except RuntimeError as exc:
            log.warning('Error in Value Tracer %d of trainer %d, exception: %s', self.replicate_rank, self.trainer_idx,
                        exc)
            log.warning('Terminate process...')
            self.terminate = True
        except KeyboardInterrupt:
            log.warning('Keyboard interrupt detected on Value Tracer %d of trainer %d', self.replicate_rank,
                        self.trainer_idx)
            self.terminate = True
        except Exception:
            log.exception('Unknown exception in Value Tracer %d of trainer %d', self.replicate_rank, self.trainer_idx)
            self.terminate = True

        time.sleep(0.1)
        log.info('Value Tracer %d of trainer %d timing: %s', self.replicate_rank, self.trainer_idx, timing)

    def close(self):
        self.task_queue.put(TaskType.TERMINATE)
