import time
from envs.env_wrappers import ShareDummyVecEnv
import numpy as np


class Actor:
    def __init__(self, actor_id, env_fn, agent_rref, args):
        self.id = actor_id
        self.num_split = num_split = args.num_split
        self.env_per_actor = env_per_actor = args.env_per_actor
        assert env_per_actor % num_split == 0
        self.env_per_split = env_per_actor // num_split

        self.verbose_time = args.verbose_time

        self.episode_length = args.episode_length
        self.agent_rref = agent_rref

        self.envs = []
        for i in range(self.num_split):
            self.envs.append(
                ShareDummyVecEnv([
                    lambda: env_fn(self.id * self.env_per_actor + i * self.env_per_split + j, args)
                    for j in range(self.env_per_split)
                ]))
        self.action_futures = []
        print('-' * 8 + ' Actor {} set up successfully! '.format(self.id) + '-' * 8)

    def run(self):
        for i, env in enumerate(self.envs):
            model_inputs = env.reset()
            # obs, state, avail_action
            assert len(model_inputs) == 3
            action_fut = self.agent_rref.rpc_async().select_action(self.id, i, model_inputs, init=True)
            self.action_futures.append(action_fut)
        while True:
            wait_times = []
            delays = []
            for i, env in enumerate(self.envs):
                wait_tik = time.time()
                delay_tik, actions = self.action_futures[i].wait()
                tok = time.time()
                wait_times.append((tok - wait_tik) * 1e3)
                delays.append((tok - delay_tik) * 1e3)
                model_inputs = env.step(actions)
                # obs, state, reward, done, infos, avail_action
                assert len(model_inputs) == 6
                self.action_futures[i] = self.agent_rref.rpc_async().select_action(self.id, i, model_inputs)
            if self.verbose_time:
                print('actor {} wait time {}ms, delay {}ms'.format(self.id, np.mean(wait_times), np.mean(delays)))
