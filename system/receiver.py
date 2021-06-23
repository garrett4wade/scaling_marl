import signal
import time
from queue import Empty

import torch
import multiprocessing as mp

from utils.timing import Timing
from utils.utils import log, join_or_kill, TaskType
import zmq
import blosc
from collections import deque
import numpy as np


class Receiver:
    def __init__(self, cfg, idx, task_queue, buffer):
        self.cfg = cfg
        self.receiver_idx = idx
        self.buffer = buffer

        self.task_queue = task_queue

        self.socket = None

        self.receiving_intervals = deque([], maxlen=100)
        self.last_recv_time = None
        self.last_send_time = None

        self.terminate = False

        self.initialized = False

        self.process = mp.Process(target=self._run)
        self.process.start()

    def _init(self):
        self.socket = zmq.Context().socket(zmq.ROUTER)
        seg_port = self.cfg.seg_addrs[self.receiver_idx].split(':')[-1]
        self.socket.bind('tcp://*:' + seg_port)

        self.initialized = True
        log.info('Reiceiver %d is ready!', self.receiver_idx)

    def _unpack_msg(self, timing, msg):
        msg = msg[2:]
        assert len(msg) % 2 == 0

        seg_dict = {}
        decompression_time = 0
        for i in range(len(msg) // 2):
            k, v = msg[2 * i].decode('ascii'), msg[2 * i + 1]
            shape, dtype = self.buffer.shapes_and_dtypes[k]

            tik = time.time()
            with timing.add_time('decompression'):
                decompressed = blosc.decompress(v)
            decompression_time += time.time() - tik

            array = np.frombuffer(decompressed, dtype=np.float32).reshape(*shape)
            seg_dict[k] = array

        tik = time.time()
        with timing.add_time('put_buffer'):
            self.buffer.put(seg_dict)
        buffer_put_time = time.time() - tik

        log.info('Receiver {} decompression time: {:.2f}, buffer put time: {:.2f}'.format(
            self.receiver_idx, decompression_time, buffer_put_time))

    def _run(self):
        log.info('Initializing Receiver %d...', self.receiver_idx)

        # should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        self.init()
        while not self.terminate:
            try:
                # receive INIT and TERMINATE signal from the main process
                try:
                    task_type = self.task_queue.get_nowait()

                    # task from the task_queue
                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        self.terminate = True
                        break

                    self.task_queue.task_done()
                except Empty:
                    pass

                if self.initialized:
                    try:
                        # we don't care what we receive from the head node
                        msg = self.socket.recv_multipart(flags=zmq.NOBLOCK)

                        self.socket.send_multipart([msg[0], msg[1], b'ok'])

                        if len(msg) > 3:
                            if self.last_recv_time:
                                self.receiving_intervals.append(time.time() - self.last_recv_time)
                            self.last_recv_time = time.time()

                            self._unpack_msg(timing, msg)
                            log.info('Receiver %d receives data from worker node %s...', self.receiver_idx, msg[0])
                    except zmq.ZMQError:
                        pass

            except RuntimeError as exc:
                log.warning('Error while transmitting data tran: %d, exception: %s', self.receiver_idx, exc)
                log.warning('Terminate process...')
                self.terminate = True
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on Receiver %d', self.receiver_idx)
                self.terminate = True
            except Exception:
                log.exception('Unknown exception in Receiver')
                self.terminate = True

        self.socket.close()
        time.sleep(0.2)
        log.info('Receiver avg. receiving interval: %.2f, timing: %s', np.mean(self.receiving_intervals), timing)

    def init(self):
        self.task_queue.put(TaskType.INIT)

    def close(self):
        self.task_queue.put(TaskType.TERMINATE)

    def join(self):
        join_or_kill(self.process)
