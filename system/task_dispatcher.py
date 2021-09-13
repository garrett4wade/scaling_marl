import zmq
import itertools
import torch.multiprocessing as mp
import time
import os
import datetime
import pathlib
import wandb
import yaml
import numpy as np
from tensorboardX import SummaryWriter
from utils.utils import log, join_or_kill, TaskType


class TaskDispatcher:
    def __init__(self, cfg, meta_controller):
        self.cfg = cfg

        self.task_socket = None

        # initialize save dir
        self.summary_dir = self.cfg.summary_dir
        if self.summary_dir is None:
            self.summary_dir = pathlib.Path('./logs')
        else:
            self.summary_dir = pathlib.Path(self.summary_dir)
        self.summary_dir /= self.cfg.env_name
        if self.cfg.env_name == 'StarCraft2':
            self.summary_dir /= self.cfg.map_name
        self.summary_dir = self.summary_dir / cfg.algorithm_name / cfg.experiment_name
        if not self.summary_dir.exists():
            os.makedirs(str(self.summary_dir))

        self.no_summary = self.cfg.no_summary
        self.use_wandb = self.cfg.use_wandb

        self.use_eval = self.cfg.use_eval
        self.eval_interval = self.cfg.eval_interval

        # TODO: finish meta controller
        from meta_controllers.naive import NaiveMetaController
        self.meta_controller = meta_controller
        assert isinstance(self.meta_controller, NaiveMetaController)

        self.terminate = False

        self.num_learner_tasks = cfg.num_policies
        self.num_worker_tasks = len(cfg.seg_addrs[0]) * cfg.num_tasks_per_node

        self.worker_socket_ident = [None for _ in range(self.num_worker_tasks)]
        self.policy_id2task_ident = {}

        self.learner_socket_ident = [None for _ in range(self.num_learner_tasks)]

        self.consumed_num_steps = [0 for _ in range(self.cfg.num_policies)]
        self.policy_version = [0 for _ in range(self.cfg.num_policies)]
        self.training_tik = None

        self.train_for_env_steps = self.cfg.train_for_env_steps
        self.train_for_seconds = self.cfg.train_for_seconds
        self.transitions_per_batch = (self.cfg.episode_length * self.cfg.actor_group_size * self.cfg.envs_per_actor *
                                      self.cfg.slots_per_update // self.cfg.num_splits)

        num_all_trainers = 0
        for _, local_config in self.cfg.learner_config.items():
            num_all_trainers += len(local_config)
        # just a estimation, it is incorrect if different policies occpy different number of GPUs
        self.train_for_episodes = self.train_for_env_steps // self.transitions_per_batch // (num_all_trainers //
                                                                                             self.cfg.num_policies)

        self._context = None
        self.task_socket = None
        self.result_socket = None

        self.accumulated_too_much_experience = [False for _ in range(self.cfg.num_policies)]
        self.accumulated_too_few_experience = [True for _ in range(self.cfg.num_policies)]
        self.stop_experience_collection = [False for _ in range(self.cfg.num_policies)]

        self.process = mp.Process(target=self._run)

    def _init_summary(self):
        algo = self.cfg.algorithm_name
        network_cls = 'rnn' if algo == 'rmappo' else 'mlp' if algo == 'mappo' else None
        postfix = 'r{}_'.format(str(self.cfg.sample_reuse)) + network_cls
        exp_name = str(self.cfg.experiment_name) + "_seed" + str(self.cfg.seed)
        if self.cfg.use_wandb:
            self.run = wandb.init(config=self.cfg,
                                  project=self.cfg.project_name,
                                  entity=self.cfg.user_name,
                                  name=exp_name,
                                  group=self.cfg.group_name,
                                  dir=str(self.summary_dir),
                                  job_type="training",
                                  reinit=True)
        else:
            curr_run = exp_name + postfix + '_' + str(datetime.datetime.now()).replace(' ', '_')
            self.summary_dir /= curr_run
            if not self.summary_dir.exists():
                os.makedirs(str(self.summary_dir))
            self.writter = SummaryWriter(self.summary_dir)

    def start_process(self):
        self.process.start()

    def _init(self):
        self._context = zmq.Context()

        self.task_socket = self._context.socket(zmq.PUB)
        task_dispatcher_port = self.cfg.task_dispatcher_addr.split(':')[-1]
        self.task_socket.bind('tcp://*:' + task_dispatcher_port)

        self.result_socket = self._context.socket(zmq.PULL)
        result_port = self.cfg.task_result_addr.split(':')[-1]
        self.result_socket.bind('tcp://*:' + result_port)

        ready_tasks = 0
        while ready_tasks < self.num_learner_tasks + self.num_worker_tasks:
            msg = self.result_socket.recv_multipart()
            idx = int(msg[0].decode('ascii').split('-')[-1])
            policy_id = int(msg[1].decode('ascii'))

            if 'learner' in msg[0].decode('ascii'):
                self.learner_socket_ident[idx] = msg[0]
            elif 'worker' in msg[0].decode('ascii'):
                self.worker_socket_ident[idx] = msg[0]
                if policy_id not in self.policy_id2task_ident:
                    self.policy_id2task_ident[policy_id] = [msg[0]]
                else:
                    self.policy_id2task_ident[policy_id].append(msg[0])
            else:
                raise NotImplementedError

            ready_tasks += 1

        assert all(self.worker_socket_ident) and all(self.learner_socket_ident), (self.worker_socket_ident,
                                                                                  self.learner_socket_ident)

        if not self.cfg.no_summary:
            self._init_summary()

        config_file = open(os.path.join(str(self.summary_dir), 'config.yaml'), 'w')
        yaml.dump(vars(self.cfg), config_file)
        config_file.close()

        tasks = self.meta_controller.reset(self.learner_socket_ident, self.worker_socket_ident)
        assert len(tasks) >= self.num_learner_tasks + self.num_worker_tasks
        for task in tasks:
            self.task_socket.send_multipart(task)

        self.training_tik = time.time()

    def _should_end_training(self):
        end = all([c_step > self.train_for_env_steps for c_step in self.consumed_num_steps])
        end |= all([v > self.train_for_episodes for v in self.policy_version])
        end |= (time.time() - self.training_tik) > self.train_for_seconds

        # if self.cfg.benchmark:
        #     end |= self.total_env_steps_since_resume >= int(2e6)
        #     end |= sum(self.samples_collected) >= int(1e6)

        return end

    def report(self, msg):
        # msg format: identity, policy_id, *summary_keys, summary_data (numpy array)
        policy_id = int(msg[1].decode('ascii'))
        data = np.frombuffer(memoryview(msg[-1]), dtype=np.float32)
        infos = {}
        for i, key in enumerate(msg[2:-1]):
            infos[key.decode('ascii')] = data[i]
        assert len(data) == len(msg[2:-1])

        if 'learner' in msg[0].decode('ascii'):
            policy_version = infos['iteration']
            self.policy_version[policy_id] = policy_version
            self.consumed_num_steps[policy_id] = policy_version * self.transitions_per_batch
            self.accumulated_too_much_experience[policy_id] = infos['buffer_util'] >= 1.0
            self.accumulated_too_few_experience[policy_id] = infos['buffer_util'] <= 0.25

        if 'workertask' in msg[0].decode('ascii'):
            log.info('Evaluation Results: %s', infos)

        if not self.no_summary:
            if self.use_wandb:
                infos = {'policy_' + str(policy_id) + '/' + k: v for k, v in infos.items()}
                wandb.log(infos, step=int(self.consumed_num_steps[policy_id].item()))
            else:
                self.writter.add_scalars('policy_' + str(policy_id),
                                         infos,
                                         step=int(self.consumed_num_steps[policy_id].item()))

    def _run(self):
        log.info('Initializing Task Dispatcher...')

        self._init()

        try:
            while not self._should_end_training():
                try:
                    msg = self.result_socket.recv_multipart(flags=zmq.NOBLOCK)
                    self.report(msg)

                    for policy_id in range(self.cfg.num_policies):
                        if self.accumulated_too_much_experience[
                                policy_id] and not self.stop_experience_collection[policy_id]:
                            for ident in self.policy_id2task_ident[policy_id]:
                                task = [ident, str(TaskType.PAUSE).encode('ascii')]
                                self.task_socket.send_multipart(task)
                            self.stop_experience_collection[policy_id] = True

                        if (self.stop_experience_collection[policy_id]
                                and self.accumulated_too_few_experience[policy_id]):
                            for ident in self.policy_id2task_ident[policy_id]:
                                task = [ident, str(TaskType.RESUME).encode('ascii')]
                                self.task_socket.send_multipart(task)
                            self.stop_experience_collection[policy_id] = False

                    new_tasks = self.meta_controller.step(msg)
                    for new_task in new_tasks:
                        self.task_socket.send_multipart(new_task)

                except zmq.ZMQError:
                    pass

        except RuntimeError:
            log.warning('Error while distributing tasks on Task Dispatcher')
            log.warning('Terminate process...')
            self.terminate = True
        except KeyboardInterrupt:
            log.warning('Keyboard interrupt detected on Task Dispatcher')
            self.terminate = True
        except Exception:
            log.exception('Unknown exception on Task Dispatcher')
            self.terminate = True

        # send termination signal to all workers and learners
        for ident in itertools.chain(self.learner_socket_ident, self.worker_socket_ident):
            msg = [ident, str(TaskType.TERMINATE).encode('ascii')]
            self.task_socket.send_multipart(msg)

        time.sleep(1)
        self.task_socket.close()
        time.sleep(0.2)
        log.info('Task Dispatcher terminated!')

    def join(self):
        join_or_kill(self.process)

    def close(self):
        pass
