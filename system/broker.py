import zmq
import itertools
import pyarrow as pa
import time
import numpy as np
import multiprocessing as mp
import threading
import queue


class Broker:
    def __init__(self, broker_id, buffer, args):
        self.id = broker_id
        self.buffer = buffer

        self.context = zmq.Context()
        self.frontend = self.context.socket(zmq.ROUTER)
        if self.id == 0:
            self.frontend.bind(args.frontend_addr)
        else:
            self.frontend.connect(args.frontend_addr)

        self.backend = self.context.socket(zmq.ROUTER)
        if self.id == 0:
            self.backend.bind(args.backend_addr)
        else:
            self.backend.connect(args.backend_addr)

        self.poller = zmq.Poller()
        self.poller.register(self.backend, zmq.POLLIN)

        # TODO: auto-tune rollout batch size
        self.rollout_bs = args.rollout_bs

        self.available_servers = []
        self.ready_clients = queue.Queue(maxsize=args.num_actors * args.num_split)

    def send_action(self):
        while True:
            client = self.ready_clients.get()
            actions = self.buffer.get_actions(client)
            data_serialized = pa.serialize((time.time(), actions)).to_buffer()
            self.frontend.send_multipart([client, b'', data_serialized])
            
    def run(self):
        backend_ready = False
        requests = []
        send_job = threading.Thread(target=self.send_action)
        send_job.start()

        while True:
            sockets = dict(self.poller.poll())

            tik1 = time.time()
            if self.backend in sockets:
                msg = self.backend.recv_multipart()
                server, empty, client0 = msg[:3]
                self.available_servers.append(server)
                if len(self.available_servers) > 0 and not backend_ready:
                    self.poller.register(self.frontend, zmq.POLLIN)
                    backend_ready = True
                if client0 != b'READY' and len(msg) > 3:
                    # this indicates server should send a reply back to clients
                    # while a 'READY' reply indicates the initialization of the server
                    clients = list(filter(lambda x: x != b'', msg))[1:-1]
                    assert len(clients) == self.rollout_bs
                    for client in clients:
                        self.ready_clients.put(client)
            # print("broker part1 time: {}".format(time.time() - tik1))

            tik2 = time.time()
            if self.frontend in sockets:
                client_addr, empty, request = self.frontend.recv_multipart()
                step_ret = pa.deserialize(request)
                if len(step_ret) == 3:
                    # after env reset
                    obs, share_obs, available_actions = step_ret
                    bs, n_ag = obs.shape[:2]
                    rewards = np.zeros((bs, n_ag, 1), dtype=np.float32)
                    dones = np.zeros_like(rewards).astype(np.bool)
                    infos = None
                else:
                    obs, share_obs, rewards, dones, infos, available_actions = step_ret
                self.buffer.insert_before_inference(client_addr, obs, share_obs, rewards, dones, infos, available_actions)
                # with self.buffer.request_lock:
                requests.append(client_addr)
            # print("broker part2 time: {}".format(time.time() - tik2))

            tik3 = time.time()
            clients = []
            # with self.buffer.request_lock:
            if len(requests) >= self.rollout_bs and len(self.available_servers) > 0:
                server = self.available_servers.pop(0)
                clients, requests = requests[:self.rollout_bs], requests[self.rollout_bs:]

            if clients:
                # TODO: the tail 'ok' is just a indicator, for debug only
                msg = [server] + list(itertools.chain(*[(b'', client) for client in clients])) + [b'', b'ok']
                self.backend.send_multipart(msg)

                if len(self.available_servers) == 0:
                    self.poller.unregister(self.frontend)
                    backend_ready = False
            # print("broker part3 time: {}".format(time.time() - tik3))
