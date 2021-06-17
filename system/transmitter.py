import signal
import time
from queue import Empty

import torch
import multiprocessing as mp

from utils.timing import Timing
from utils.utils import log, join_or_kill, TaskType, SocketState
import zmq
from faster_fifo import Queue
from collections import deque
import numpy as np
import blosc
from sys import getsizeof


class Transmitter:
    def __init__(self, cfg, idx, task_queue, buffer):
        self.cfg = cfg
        self.transmitter_idx = idx
        self.buffer = buffer

        self.task_queue = task_queue

        self.socket = None
        self.socket_state = SocketState.RECV

        self.seg_queue = Queue()

        self.sending_delays = deque([], maxlen=100)
        self.sending_intervals = deque([], maxlen=100)
        self.last_recv_time = None
        self.last_send_time = None

        self.terminate = False

        self.initialized = False

        self.process = mp.Process(target=self._run)
        self.process.start()

    def _init(self):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(self.cfg.seg_addrs[self.cfg.node_idx])
        self.socket.send(b'ready')
        self.socket_state = SocketState.SEND

        self.initialized = True

    def _pack_msg(self, slot):
        msg = []
        mem_data = {}
        total_mem = 0
        for k, data in self.buffer.storage.items():
            # lz4 is the most cost-efficient compression choice, ~7.5x compression in ~1.5s
            compressed = blosc.compress(data[slot].tobytes(), typesize=4, cname='lz4')
            mem = 
            mem_data[k] = mem = getsizeof(compressed) / 1024**2
            total_mem += mem
            msg.extend([k.encode('ascii'), compressed])
        log.info('seg size:  {} (MB), total {:.2f} MB'.format(mem_data, total_mem))
        self.socket.send_multipart(msg)
        self.last_send_time = time.time()
        self.sending_intervals.append(self.last_send_time - self.last_recv_time)
        self.socket_state = SocketState.SEND
        log.info('Successfully sending data to head node on Transmitter %d...', self.transmitter_idx)
        log.info('Remaining segs in queue: %d', self.seg_queue.qsize())

    def _run(self):
        log.info('Initializing Transmitter %d...', self.transmitter_idx)

        # should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        min_num_segs = 4

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

                if self.initialized and self.socket_state == SocketState.SEND:
                    try:
                        # we don't care what we receive from the head node
                        _ = self.socket.recv(flags=zmq.NOBLOCK)
                        self.last_recv_time = time.time()
                        if self.last_send_time:
                            self.sending_delays.append(self.last_recv_time - self.last_send_time)
                        self.socket_state = SocketState.RECV
                        log.info('Receiving data request from head node on Transmitter %d...', self.transmitter_idx)
                    except zmq.ZMQError:
                        pass

                if self.initialized and self.socket_state == SocketState.RECV:
                    try:
                        slot = self.seg_queue.get(block=False)
                    except Empty:
                        slot = None

                    if slot is not None:
                        with timing.add_time('pack_seg'):
                            self._pack_msg(slot)

                        with timing.add_time('after_sending'):
                            self.buffer.close_out(slot)

                with timing.add_time('waiting_and_prefetching'):
                    if self.seg_queue.qsize() <= min_num_segs:
                        slot = self.buffer.get(timeout=0.02)

                        if slot is not None:
                            self.seg_queue.put(slot)

            except RuntimeError as exc:
                log.warning('Error while transmitting data tran: %d, exception: %s', self.transmitter_idx, exc)
                log.warning('Terminate process...')
                self.terminate = True
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on Transmitter %d', self.transmitter_idx)
                self.terminate = True
            except Exception:
                log.exception('Unknown exception in Transmitter')
                self.terminate = True

        self.socket.close()
        time.sleep(0.2)
        log.info('Transmitter avg. sending interval: %.2f, avg. delay: %.3f, timing: %s', np.mean(self.sending_intervals), np.mean(self.sending_delays), timing)

    def init(self):
        self.task_queue.put(TaskType.INIT)

    def close(self):
        self.task_queue.put(TaskType.TERMINATE)

    def join(self):
        join_or_kill(self.process)
