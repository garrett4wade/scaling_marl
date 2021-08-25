import time
from queue import Empty

import torch
import torch.multiprocessing as mp

from utils.timing import Timing
from utils.utils import log, join_or_kill, TaskType, SocketState
import zmq
from faster_fifo import Queue
from collections import deque
import numpy as np
import blosc
from sys import getsizeof


class Transmitter:
    def __init__(self, cfg, buffers):
        self.cfg = cfg
        self.buffers = buffers

        self.termination_queue = mp.JoinableQueue()

        self.num_learner_nodes = len(self.cfg.learner_config)

        self._context = None
        self.sockets = [None for _ in range(self.num_learner_nodes)]
        self.socket_states = [SocketState.RECV for _ in range(self.num_learner_nodes)]

        self.seg_queues = [Queue() for _ in range(self.num_learner_nodes)]

        self.sending_delays = deque([], maxlen=100)
        self.sending_intervals = deque([], maxlen=100)
        self.last_recv_time = [None for _ in range(self.num_learner_nodes)]
        self.last_send_time = [None for _ in range(self.num_learner_nodes)]

        self.terminate = False
        self.initialized = False

        self.remaining_segs = 0

        self.process = mp.Process(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        self._context = zmq.Context()
        for i in range(self.num_learner_nodes):
            socket = self._context.socket(zmq.REQ)
            socket.connect(self.cfg.seg_addrs[i][self.cfg.worker_node_idx])
            socket.identity = ("node-" + str(self.cfg.worker_node_idx) + "to" + str(i)).encode('ascii')
            self.sockets[i] = socket

        for i, socket in enumerate(self.sockets):
            socket.send(b'ready')
            self.socket_states[i] = SocketState.SEND

        self.initialized = True

    def _pack_msg(self, buffer_id, slot, dst):
        learner_node_idx, local_trainer_idx = dst

        msg = []
        mem_data = {}
        total_mem = 0
        num_valid_agents = int(self.buffers[buffer_id].mask_aa_obs_spoof[slot, 0, 0, 0].sum().item()) + 1
        msg.append(str(num_valid_agents).encode('ascii'))
        tmp_valid_agents = self.buffers[buffer_id].mask_aa_obs_spoof[slot, np.random.randint(0, self.cfg.episode_length), np.random.randint(0, self.cfg.envs_per_actor // self.cfg.num_splits), np.random.randint(0, num_valid_agents)].sum()
        assert tmp_valid_agents == num_valid_agents - 1, (tmp_valid_agents, num_valid_agents, self.buffers[buffer_id].mask_aa_obs_spoof[slot, 0, 0])
        for k, data in self.buffers[buffer_id].storage.items():
            valid_data = data[slot, :, :, :num_valid_agents]
            valid_data = valid_data.reshape(valid_data.shape[0], -1, *valid_data.shape[3:])
            # lz4 is the most cost-efficient compression choice, ~7.5x compression in ~1.5s in 27m_vs_30m env
            compressed = blosc.compress(valid_data.tobytes(), typesize=4, cname='lz4')
            mem_data[k] = mem = getsizeof(compressed) / 1024**2
            total_mem += mem
            msg.extend([k.encode('ascii'), compressed])

        with self.buffers[buffer_id].env_summary_lock:
            summary_info = self.buffers[buffer_id].summary_block.sum(axis=(0, 1))
        msg.extend([
            self.sockets[learner_node_idx].identity, summary_info,
            str(buffer_id).encode('ascii'),
            str(local_trainer_idx).encode('ascii')
        ])

        # log.info('seg size:  {} (MB), total {:.2f} MB'.format(mem_data, total_mem))
        self.sockets[learner_node_idx].send_multipart(msg)

        self.last_send_time[learner_node_idx] = time.time()
        self.sending_intervals.append(self.last_send_time[learner_node_idx] - self.last_recv_time[learner_node_idx])
        self.socket_states[learner_node_idx] = SocketState.SEND

        # log.info('Successfully sending data to head node on Transmitter...')
        # log.info('Remaining segs in queue: %d', self.remaining_segs)

    def _run(self):
        log.info('Initializing Transmitter...')

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        min_num_segs = 4

        self._init()
        while not self.terminate:
            try:
                # receive TERMINATE signal from the main process
                try:
                    task_type = self.termination_queue.get_nowait()
                    if task_type == TaskType.TERMINATE:
                        self.terminate = True
                        break
                    self.termination_queue.task_done()
                except Empty:
                    pass

                for i, socket in enumerate(self.sockets):
                    if self.initialized and self.socket_states[i] == SocketState.SEND:
                        try:
                            # we don't care what we receive from the head node
                            _ = socket.recv(flags=zmq.NOBLOCK)
                            self.last_recv_time[i] = time.time()
                            if self.last_send_time[i] is not None:
                                self.sending_delays.append(self.last_recv_time[i] - self.last_send_time[i])
                            self.socket_states[i] = SocketState.RECV
                            # log.info('Receiving data request from head node on Transmitter...')
                        except zmq.ZMQError:
                            pass

                    if self.initialized and self.socket_states[i] == SocketState.RECV:
                        try:
                            buffer_id, slot, local_trainer_dst = self.seg_queues[i].get(block=False)
                        except Empty:
                            buffer_id, slot, local_trainer_dst = None, None, None

                        if slot is not None:
                            with timing.add_time('pack_seg'):
                                self._pack_msg(buffer_id, slot, (i, local_trainer_dst))

                            with timing.add_time('after_sending'):
                                self.buffers[buffer_id].close_out(slot)
                                self.remaining_segs -= 1

                with timing.add_time('waiting_and_prefetching'):
                    if self.remaining_segs <= min_num_segs:
                        for buffer_id, buffer in enumerate(self.buffers):
                            slot, (learner_node_idx, local_trainer_dst) = buffer.get(timeout=0.02)

                            if slot is not None:
                                self.seg_queues[learner_node_idx].put((buffer_id, slot, local_trainer_dst))
                                self.remaining_segs += 1

            except RuntimeError as exc:
                log.warning('Error while transmitting data, exception: %s', exc)
                log.warning('Terminate process...')
                self.terminate = True
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on Transmitter')
                self.terminate = True
            except Exception:
                log.exception('Unknown exception in Transmitter')
                self.terminate = True

        for socket in self.sockets:
            socket.close()
        time.sleep(0.2)
        log.info('Transmitter avg. sending interval: %.2f, avg. delay: %.3f, timing: %s',
                 np.mean(self.sending_intervals), np.mean(self.sending_delays), timing)

    def close(self):
        self.termination_queue.put(TaskType.TERMINATE)

    def join(self):
        join_or_kill(self.process)
