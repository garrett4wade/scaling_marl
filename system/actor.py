from envs.env_wrappers import ShareDummyVecEnv


class Actor:
    def __init__(self, actor_id, env_fn, agent_rref, args):
        self.id = actor_id
        self.num_split = num_split = args.num_split
        self.env_per_actor = env_per_actor = args.env_per_actor
        assert env_per_actor % num_split == 0
        self.env_per_split = env_per_actor // num_split

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

    def run(self):
        for i, env in enumerate(self.envs):
            model_inputs = env.reset()
            # obs, state, avail_action
            assert len(model_inputs) == 3
            action_fut = self.agent_rref.rpc_async().select_action(self.id, i, model_inputs, init=True)
            self.action_futures.append(action_fut)
        while True:
            for i, env in enumerate(self.envs):
                actions = self.action_futures[i].wait()
                model_inputs = env.step(actions)
                # obs, state, reward, done, infos, avail_action
                assert len(model_inputs) == 6
                self.action_futures[i] = self.agent_rref.rpc_async().select_action(self.id, i, model_inputs)
