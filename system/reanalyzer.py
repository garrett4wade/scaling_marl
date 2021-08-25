import psutil
from utils.timing import Timing
import time

from queue import Empty
from utils.utils import log, TaskType

import torch
import torch.multiprocessing as mp


class Reanalyzer:
    def __init__(
        self,
        cfg,
        trainer_idx,
        replicate_rank,
        gpu_rank,
        buffer,
        value_tracer_queue,
        batch_queue,
        task_queue,
        shm_state_dict,
        trainer_policy_version,
        param_lock,
    ):
        self.cfg = cfg
        self.trainer_idx = trainer_idx
        self.replicate_rank = replicate_rank
        self.gpu_rank = gpu_rank

        self.buffer = buffer
        self.policy_id = buffer.policy_id
        self.reanalyzer_idx = self.trainer_idx * cfg.num_reanalyzers_per_trainer + self.replicate_rank

        self.num_agents = len(self.cfg.policy2agents[str(self.policy_id)])
        example_agent = self.cfg.policy2agents[str(self.policy_id)][0]

        self.obs_space = self.cfg.observation_space[example_agent]
        self.act_space = self.cfg.action_space[example_agent]

        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        # policy network
        self.policy_fn = Policy

        # to send reanalyzed data to value_tracer
        self.value_tracer_queue = value_tracer_queue
        # to synchronize weights with trainers
        self.shm_state_dict = shm_state_dict
        self.trainer_policy_version = trainer_policy_version
        self.local_policy_version = -1
        assert self.trainer_policy_version.is_shared()
        for k, v in self.shm_state_dict.items():
            assert v.is_shared()
        self.param_lock = param_lock

        self.batch_queue = batch_queue
        self.task_queue = task_queue

        self.initialized = self.terminate = False

        self.process = mp.Process(target=self._run)

    def start_process(self):
        self.process.start()

    def _init(self, timing):
        assert self.cfg.cuda and torch.cuda.is_available(), 'cpu training currently not supported'
        torch.cuda.set_device(self.gpu_rank)

        # policy network
        self.policy = self.policy_fn(self.gpu_rank, self.cfg, self.obs_space, self.act_space, is_training=False)
        self.policy.eval_mode()

        log.debug('Reanalyzer {} of trainer {} waiting for all nodes ready...'.format(
            self.replicate_rank, self.trainer_idx))

        with self.param_lock.r_locked():
            with timing.time_avg('update_weights/load_state_dict_once'):
                self.policy.load_state_dict(self.shm_state_dict)
                self.local_policy_version = self.trainer_policy_version.item()

            log.info('Reanalyzer %d of trainer %d --- Update to policy version %d', self.replicate_rank,
                     self.trainer_idx, self.local_policy_version)
        self.initialized = True

    def maybe_update_weights(self, timing):
        with self.param_lock.r_locked():
            if (self.local_policy_version + self.cfg.sample_reuse * self.cfg.broadcast_interval <=
                    self.trainer_policy_version):
                with timing.time_avg('update_weights/load_state_dict_once'):
                    self.policy.load_state_dict(self.shm_state_dict)
                    self.local_policy_version = self.trainer_policy_version.item()

                log.info('Reanalyzer %d of trainer %d --- Update to policy version %d', self.replicate_rank,
                         self.trainer_idx, self.local_policy_version)

    @torch.no_grad()
    def reanalyze_step(self, slot_id, timing):
        if self.cfg.use_reanalyze:
            self.maybe_update_weights(timing)

        with timing.add_time('burn_in'):
            # you can also conduct burn-in like in R2D2 here
            pass

        with timing.add_time('reanalyze_value'):
            # re-compute values/rnn_states for learning (re-analysis in MuZero, burn-in in R2D2 etc.)
            # TODO: deal with Hanabi (nonshared case)
            if self.cfg.use_reanalyze:
                share_obs = self.buffer.share_obs[slot_id]
                reanalyze_inputs = {
                    'share_obs': share_obs.reshape(self.cfg.episode_length + 1, -1, *share_obs.shape[3:])
                }
                if self.cfg.use_recurrent_policy:
                    rnn_states_critic = self.buffer.rnn_states_critic[slot_id][0]
                    rnn_states_critic = rnn_states_critic.reshape(-1, *rnn_states_critic.shape[2:]).swapaxes(0, 1)

                    masks = self.buffer.masks[slot_id]
                    masks = masks.reshape(self.cfg.episode_length + 1, -1, *masks.shape[3:])
                    reanalyze_inputs = {
                        **reanalyze_inputs,
                        'rnn_states_critic': rnn_states_critic,
                        'masks': masks,
                    }
                for k, v in reanalyze_inputs.items():
                    reanalyze_inputs[k] = torch.from_numpy(v).to(**self.tpdv)

                values = self.policy.get_values(**reanalyze_inputs).cpu().numpy()
                self.buffer.values[slot_id] = values.reshape(*self.buffer.values[slot_id].shape)

        with timing.add_time('compute_importance_weights'):
            # also compute data used in value tracer, e.g. importance weights for V-trace
            # by default we use GAE, thus we do nothing here
            pass

    def _run(self):
        psutil.Process().nice(self.cfg.default_niceness + 3)

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        # this minimal requirement ensures there are at most 1 slot per tracer in the queue
        # waiting for trace computation
        min_num_waiting_slots = 1
        # waiting in queues + # reanalyzer + # tracer + 1 in trainer + 1 readable + 1 being written
        assert (self.cfg.num_value_tracers_per_trainer + self.cfg.num_reanalyzers_per_trainer +
                2 < self.cfg.qsize), 'please increase qsize!'

        self._init(timing)

        try:
            while not self.terminate:

                if self.value_tracer_queue.qsize() + self.batch_queue.qsize() < min_num_waiting_slots:
                    with timing.add_time('wait_for_batch'):
                        slot_id = self.buffer.get(timeout=0.5)

                    if slot_id is not None:
                        self.reanalyze_step(slot_id, timing)

                        self.value_tracer_queue.put(slot_id)

                with timing.add_time('get_tasks'):
                    try:
                        task_type = self.task_queue.get_nowait()

                        if task_type == TaskType.TERMINATE:
                            self.terminate = True

                        self.task_queue.task_done()
                    except Empty:
                        pass

        except RuntimeError as exc:
            log.warning('Error in Reanalyzer %d of trainer %d, exception: %s', self.replicate_rank, self.trainer_idx,
                        exc)
            log.warning('Terminate process...')
            self.terminate = True
        except KeyboardInterrupt:
            log.warning('Keyboard interrupt detected on Reanalyzer %d of trainer %d', self.replicate_rank,
                        self.trainer_idx)
            self.terminate = True
        except Exception:
            log.exception('Unknown exception in Reanalyzer %d of trainer %d', self.replicate_rank, self.trainer_idx)
            self.terminate = True

        time.sleep(0.1)
        log.info('Reanalyzer %d of trainer %d timing: %s', self.replicate_rank, self.trainer_idx, timing)

    def close(self):
        self.task_queue.put(TaskType.TERMINATE)
