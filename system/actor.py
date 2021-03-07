from envs.env_wrappers import ShareDummyVecEnv


class Actor:
    def __init__(self, actor_id, env_fns, agent_rref, args):
        self.id = actor_id
        self.num_split = num_split = args.num_split
        self.env_per_actor = env_per_actor = args.env_per_actor
        assert env_per_actor % num_split == 0
        self.env_per_split = env_per_actor // num_split

        self.episode_length = args.episdoe_length
        self.agent_rref = agent_rref

        assert len(env_fns) == self.env_per_actor
        self.envs = []
        for i in range(self.num_split):
            env_slice = slice(i * self.env_per_split, (i + 1) * self.env_per_split)
            assert len(env_fns[env_slice]) == self.env_per_split
            self.envs.append(ShareDummyVecEnv(env_fns[env_slice]))
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
