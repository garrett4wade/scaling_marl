import zmq
import itertools
import multiprocessing as mp
import time
from utils.utils import log, join_or_kill, TaskType


class TaskDispatcher:
    def __init__(self, cfg, meta_controller):
        self.cfg = cfg

        self.socket = None

        # TODO: finish meta controller
        from meta_controllers.naive import NaiveMetaController
        self.meta_controller = meta_controller
        assert isinstance(self.meta_controller, NaiveMetaController)

        self.initialized = False
        self.terminate = False

        assert cfg.num_tasks_per_node == 1
        # TODO: add multiple tasks on each node
        self.num_learner_tasks = cfg.num_policies
        self.num_worker_tasks = len(cfg.seg_addrs[0]) * cfg.num_tasks_per_node
        self.ready_tasks = 0

        self.worker_socket_addrs = [None for _ in range(self.num_worker_tasks)]
        self.learner_socket_addrs = [None for _ in range(self.num_learner_tasks)]

        self.consumed_num_steps = [0 for _ in range(self.cfg.num_policies)]
        self.policy_version = [0 for _ in range(self.cfg.num_policies)]
        self.training_tik = None

        self.process = mp.Process(target=self._run)

    def start_process(self):
        self.process.start()

    def _init(self):
        self.socket = zmq.Context().socket(zmq.ROUTER)

        task_dispatcher_port = self.cfg.task_dispatcher_addr.split(':')[-1]
        self.socket.bind('tcp://*:' + task_dispatcher_port)

        while self.ready_tasks < self.num_learner_tasks + self.num_worker_tasks:
            msg = self.socket.recv_multipart()
            assert len(msg) == 3 and msg[1] == b''
            ready_msg = msg[2].decode('ascii')
            idx = int(ready_msg.split('-')[-1])

            if 'learner' in ready_msg:
                self.learner_socket_addrs[idx] = msg[0]
            elif 'worker' in ready_msg:
                self.worker_socket_addrs[idx] = msg[0]
            else:
                raise NotImplementedError

            self.ready_tasks += 1

        tasks = self.meta_controller.reset()
        for task, addr in zip(tasks, itertools.chain(self.learner_socket_addrs, self.worker_socket_addrs)):
            msg = [addr, b''] + list(task)
            self.socket.send_multipart(msg)

        self.initialized = True
        self.training_tik = time.time()

    def _should_end_training(self):
        # TODO: use messages from workers and learners to update consumed_num_steps and policy_version
        end = all([c_step > self.cfg.train_for_env_steps for c_step in self.consumed_num_steps])
        end |= all([v > self.cfg.train_for_episodes for v in self.policy_version])
        end |= (time.time() - self.training_tik) > self.train_for_seconds

        # if self.cfg.benchmark:
        #     end |= self.total_env_steps_since_resume >= int(2e6)
        #     end |= sum(self.samples_collected) >= int(1e6)

        return end

    def _run(self):
        log.info('Initializing Task Dispatcher...')

        self._init()

        try:
            while not self._should_end_training():
                if self.initialized:
                    try:
                        # we don't care what we receive from the head node
                        msg = self.socket.recv_multipart(flags=zmq.NOBLOCK)

                        # TODO: deal with messages from workers and learners
                        new_tasks = self.meta_controller.step(msg)
                        for new_task in new_tasks:
                            pass

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
        for addr in itertools.chain(self.learner_socket_addrs, self.worker_socket_addrs):
            msg = [addr, b'', str(TaskType.TERMINATE).encode('ascii')]
            self.socket.send_multipart(msg)

        time.sleep(1)
        self.socket.close()
        time.sleep(0.2)
        log.info('Task Dispatcher terminated!')

    def join(self):
        join_or_kill(self.process)
