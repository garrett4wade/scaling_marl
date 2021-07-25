import zmq
import signal
import itertools
from queue import Empty
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
        self.termination_queue = mp.JoinableQueue(1)

        assert cfg.num_tasks_per_node == 1
        # TODO: add multiple tasks on each node
        self.num_learner_tasks = cfg.num_policies
        self.num_worker_tasks = len(cfg.seg_addrs[0]) * cfg.num_tasks_per_node
        self.ready_tasks = 0

        self.worker_socket_addrs = [None for _ in range(self.num_worker_tasks)]
        self.learner_socket_addrs = [None for _ in range(self.num_learner_tasks)]

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

    def _run(self):
        log.info('Initializing Task Dispatcher...')

        # should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self._init()
        while not self.terminate:
            try:
                try:
                    _ = self.termination_queue.get_nowait()
                    self.termination_queue.task_done()
                    self.terminate = True
                    break
                except Empty:
                    pass

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

        self.socket.close()
        time.sleep(0.2)
        log.info('Task Dispatcher terminated!')

    def close(self):
        self.termination_queue.put(TaskType.TERMINATE)

    def join(self):
        join_or_kill(self.process)
