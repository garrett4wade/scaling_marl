import time
from queue import Empty

import torch
import torch.multiprocessing as mp

from utils.timing import Timing
from utils.utils import log, join_or_kill, TaskType
import zmq
import blosc
from collections import deque
import numpy as np


class Receiver:
    def __init__(self, cfg, rank, trainers, nodes_ready_event, trainer_buffer_ready_events):
        self.cfg = cfg
        self.rank = rank

        for e in trainer_buffer_ready_events:
            e.wait()

        self.buffers = [trainer.buffer for trainer in trainers]

        self.termination_queue = mp.JoinableQueue(1)

        self.socket = None
        # learner must broadcast model weights after all worker nodes have finished env.reset,
        # otherwise the initialized model weights will be thrown away by ZeroMQ, which causes a dead lock
        # as the initialization of worker nodes and learner nodes are asynchronous,
        # we must set a nodes_ready_event to indicate when to broadcast model weights
        self.nodes_ready_event = nodes_ready_event

        self.receiving_intervals = deque([], maxlen=100)
        self.last_recv_time = None
        self.last_send_time = None

        self.terminate = False

        self.recv_cnt = 0

        self.process = mp.Process(target=self._run, daemon=True)

    def start_proess(self):
        self.process.start()

    def _init(self):
        self.socket = zmq.Context().socket(zmq.ROUTER)
        seg_port = self.cfg.seg_addrs[self.cfg.learner_node_idx][self.rank].split(':')[-1]
        self.socket.bind('tcp://*:' + seg_port)

        log.info('Reiceiver %d is ready!', self.rank)

    def _unpack_msg(self, timing, msg):
        self.recv_cnt += 1
        msg = msg[2:]
        assert len(msg) % 2 == 0
        buffer = self.buffers[int(msg[-1].decode('ascii'))]

        seg_dict = {}
        decompression_time = 0
        for i in range(len(msg) // 2 - 2):
            k, v = msg[2 * i].decode('ascii'), msg[2 * i + 1]
            shape, dtype = buffer.shapes_and_dtypes[k]

            tik = time.time()
            with timing.add_time('decompression'):
                decompressed = blosc.decompress(v)
            decompression_time += time.time() - tik

            array = np.frombuffer(decompressed, dtype=np.float32).reshape(*shape)
            seg_dict[k] = array

        socket_ident, summary_info = msg[-4:-2]
        task_rank = int(msg[-2].decode('ascii'))
        summary_info = np.frombuffer(summary_info, dtype=np.float32)

        assert socket_ident.decode('ascii')[-1] == str(self.cfg.learner_node_idx)
        worker_node_idx = int(socket_ident.decode('ascii').split('-')[-1][0])

        tik = time.time()
        with timing.add_time('put_buffer'):
            buffer.put(seg_dict)

            with buffer.env_summary_lock:
                buffer.summary_block[worker_node_idx, task_rank] = summary_info
        buffer_put_time = time.time() - tik

        if self.recv_cnt % 100 == 0:
            log.info('Receiver {} decompression time: {:.2f}, buffer put time: {:.2f}'.format(
                self.rank, decompression_time, buffer_put_time))

    def _run(self):
        log.info('Initializing Receiver %d...', self.rank)

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        self._init()
        try:
            while not self.terminate:
                # receive TERMINATE signal from the main process
                try:
                    task_type = self.termination_queue.get_nowait()

                    if task_type == TaskType.TERMINATE:
                        self.terminate = True
                        break

                    self.termination_queue.task_done()
                except Empty:
                    pass

                try:
                    # we don't care what we receive from the head node
                    msg = self.socket.recv_multipart(flags=zmq.NOBLOCK)

                    self.socket.send_multipart([msg[0], msg[1], b'ok'])

                    if len(msg) > 3:
                        # this is a data message
                        if self.last_recv_time:
                            self.receiving_intervals.append(time.time() - self.last_recv_time)
                        self.last_recv_time = time.time()

                        self._unpack_msg(timing, msg)
                        # log.info('Receiver %d receives data from worker node %s...', self.rank, msg[-2])
                    else:
                        # this is a ready indicator
                        self.nodes_ready_event.set()

                except zmq.ZMQError:
                    pass

        except RuntimeError as exc:
            log.warning('Error while receiving data Receiver: %d, exception: %s', self.rank, exc)
            log.warning('Terminate process...')
            self.terminate = True
        except KeyboardInterrupt:
            log.warning('Keyboard interrupt detected on Receiver %d', self.rank)
            self.terminate = True
        except Exception:
            log.exception('Unknown exception in Receiver')
            self.terminate = True

        self.socket.close()
        time.sleep(0.2)
        log.info('Receiver avg. receiving interval: %.2f, timing: %s', np.mean(self.receiving_intervals), timing)

    def close(self):
        self.termination_queue.put(TaskType.TERMINATE)

    def join(self):
        join_or_kill(self.process)
