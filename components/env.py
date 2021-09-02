class BaseEnv:
    def reset(self):
        """ Return a dict of observations, called automatically by env wrappers when env is done.

        Returns:
            envreset_outputs (dict): All observations are numpy arrays, with shape [num_agents, *shape].

        Example Output:
            {
                'obs': {
                    'obs': np.random.randn(3, 10),
                    'shared_obs': np.random.randn(3, 10),
                    'available_actions': np.random.randn(3, 5)
                }
            }
        """
        raise NotImplementedError

    def step(self, actions):
        """ consume actions and advance one env step

        Args:
            actions (numpy.ndarray): Actions of all agents. Any type of actions is merged to one numpy array,
                                     including continuous-discrete mixed actions and multi-discrete actions.

        Returns:
            envstep_outputs (dict): Observations, rewards, dones and info, merged into one dict. Note that the tail
                                    dimension of rewards and dones is 1.

        Example Output:
            {
                'obs': {
                    'obs': np.random.randn(3, 10),
                    'shared_obs': np.random.randn(3, 10),
                    'available_actions': np.random.randn(3, 5)
                },
                'rewards': np.random.randn(3, 1),
                'dones': np.random.randint(0, 2, (3, 1)),
                'info': {
                    'some_summary_infos': 0
                }
            }
        """
        raise NotImplementedError
