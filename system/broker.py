import zmq
import itertools
import pyarrow as pa
import time
import numpy as np


class Broker:
    def __init__(self, buffer, args):
        self.buffer = buffer

        self.context = zmq.Context()
        self.frontend = self.context.socket(zmq.ROUTER)
        self.frontend.bind(args.frontend_addr)

        self.backend = self.context.socket(zmq.ROUTER)
        self.backend.bind(args.backend_addr)

        self.poller = zmq.Poller()
        self.poller.register(self.backend, zmq.POLLIN)

        # TODO: auto-tune rollout batch size
        self.rollout_bs = args.rollout_bs
        self.request_buffer = []

    def run(self):
        backend_ready = False
        available_servers = []

        while True:
            sockets = dict(self.poller.poll())

            if self.backend in sockets:
                msg = self.backend.recv_multipart()
                server, empty, client0 = msg[:3]
                available_servers.append(server)
                if len(available_servers) > 0 and not backend_ready:
                    self.poller.register(self.frontend, zmq.POLLIN)
                    backend_ready = True
                if client0 != b'READY' and len(msg) > 3:
                    # this indicates server should send a reply back to clients
                    # while a 'READY' reply indicates the initialization of the server
                    clients = list(filter(lambda x: x != b'', msg))[1:-1]
                    assert len(clients) == self.rollout_bs
                    for client in clients:
                        actions = self.buffer.get_actions(client)
                        data_serialized = pa.serialize((time.time(), actions)).to_buffer()
                        self.frontend.send_multipart([client, b'', data_serialized])

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
                self.request_buffer.append(client_addr)

            if len(self.request_buffer) >= self.rollout_bs and len(available_servers) > 0:
                server = available_servers.pop(0)
                clients, self.request_buffer = self.request_buffer[:self.rollout_bs], self.request_buffer[self.rollout_bs:]

                # TODO: the tail 'ok' is just a indicator, for debug only
                msg = [server] + list(itertools.chain(*[(b'', client) for client in clients])) + [b'', b'ok']
                self.backend.send_multipart(msg)

                if len(available_servers) == 0:
                    self.poller.unregister(self.frontend)
                    backend_ready = False
