import time
import signal
from collections import deque
from queue import Empty

import numpy as np
import psutil
import torch
import multiprocessing as mp

from utils.timing import Timing
from utils.utils import log, join_or_kill, cuda_envvars_for_policy, memory_stats, TaskType
import zmq
from collections import OrderedDict
from faster_fifo import Queue


class Transmitter:
    def __init__(self, cfg, idx, buffer):
        self.cfg = cfg
        self.transmitter_idx = idx
        self.buffer = buffer

        self.socket = None

        self.seg_queue = Queue()

        self.terminate = False

        self.process = mp.Process(target=self._run)
        self.process.start()
    
    def init(self):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(self.cfg.seg_addr)
        self.socket.send(b'ready')
    
    def _pack_msg(self, slot):
        msg = []
        for k, data in self.buffer.storage.items():
            msg.extend([k.encode('ascii'), data[slot]])
        self.socket.send_multipart(msg)
        log.info('Successfully sending data to head node on Transmitter %d...', self.transmitter_idx)
    
    def _run(self):
        log.info('Initializing Transmitter %d...', self.transmitter_idx)

        # should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        self.init()
        while not self.terminate:
            try:
                try:
                    msg = self.socket.recv(flags=zmq.NOBLOCK)
                    log.info('Receiving data request from head node on Transmitter %d...', self.transmitter_idx)
                except zmq.ZMQError:
                    msg = None
                
                if msg:
                    with timing.add_time('waiting'), timing.timeit('wait_seg'):
                        try:
                            slot = self.seg_queue.get(block=False)
                        except Empty:
                            slot = self.buffer.get(block=True)
                    
                    with timing.add_time('pack_seg'):
                        self._pack_msg(slot)
                    
                    with timing.add_time('after_sending'):
                        self.buffer.after_sending(slot)

                try:
                    slots = self.buffer.get_many(timeout=0.02)
                except RuntimeError:
                    slots = []
                
                for slot in slots:
                    self.seg_queue.put(slot)

            except RuntimeError as exc:
                log.warning('Error while transmitting data tran: %d, exception: %s', self.transmitter_idx, exc)
                log.warning('Terminate process...')
                self.terminate = True
            except KeyboardInterrupt:
                self.terminate = True
            except Exception:
                log.exception('Unknown exception in Transmitter')
                self.terminate = True

        self.socket.close()
