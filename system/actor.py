import time
from envs.env_wrappers import ShareDummyVecEnv
import numpy as np
import zmq
import pyarrow as pa


class Actor:
    def __init__(self, actor_id, env_fn, args):
        self.id = actor_id
        self.num_split = num_split = args.num_split
        self.env_per_actor = env_per_actor = args.env_per_actor
        assert env_per_actor % num_split == 0
        self.env_per_split = env_per_actor // num_split

        self.verbose_time = args.verbose_time

        self.episode_length = args.episode_length

        self.context = zmq.Context()
        self.sockets = []
        for i in range(self.num_split):
            socket = self.context.socket(zmq.REQ)
            socket.identity = "Client-{}-Split-{}".format(self.id, i).encode("ascii")
            socket.connect(args.frontend_addr)
            self.sockets.append(socket)

        self.envs = []
        for i in range(self.num_split):
            self.envs.append(
                ShareDummyVecEnv([
                    lambda: env_fn(self.id * self.env_per_actor + i * self.env_per_split + j, args)
                    for j in range(self.env_per_split)
                ]))
        print('-' * 8 + ' Actor {} set up successfully! '.format(self.id) + '-' * 8)

    def run(self):
        request_tik = []
        for i, env in enumerate(self.envs):
            step_ret = env.reset()
            # obs, state, avail_action
            assert len(step_ret) == 3
            request_tik.append(time.time())
            # TODO: compression may be unnecessary here because data seg is small
            self.sockets[i].send(pa.serialize(step_ret).to_buffer())
        while True:
            inf_times = []
            wait_times = []
            delays = []
            step_times = []
            for i, env in enumerate(self.envs):
                wait_tik = time.time()

                msg = self.sockets[i].recv()
                delay_tik, actions = pa.deserialize(msg)

                tok = time.time()

                step_ret = env.step(actions)

                step_tok = time.time()

                inf_times.append((tok - request_tik[i]) * 1e3)
                wait_times.append((tok - wait_tik) * 1e3)
                delays.append((tok - delay_tik) * 1e3)
                step_times.append((step_tok - tok) * 1e3)

                # obs, state, reward, done, infos, avail_action
                assert len(step_ret) == 6

                request_tik[i] = time.time()
                self.sockets[i].send(pa.serialize(step_ret).to_buffer())

            if self.verbose_time:
                print('actor {} inference time {:.2f}ms, env step time {:.2f}ms, wait time {:.2f}ms, delay {:.2f}ms'.
                      format(self.id, np.mean(inf_times), np.mean(step_times), np.mean(wait_times), np.mean(delays)))
