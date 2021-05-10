import gym
import torch
from utils.util import get_shape_from_obs_space, get_shape_from_act_space
from collections import namedtuple


def to_numpy_type(torch_type):
    return torch.zeros((), dtype=torch_type).numpy().dtype


StorageSpec = namedtuple('StorageSpec', ['name', 'shape', 'dtype', 'bootstrap', 'init_value'])


def get_ppo_storage_specs(args, obs_space, share_obs_space, act_space):
    obs_shape = get_shape_from_obs_space(obs_space)
    share_obs_shape = get_shape_from_obs_space(share_obs_space)

    if type(obs_shape[-1]) == list:
        obs_shape = obs_shape[:1]

    if type(share_obs_shape[-1]) == list:
        share_obs_shape = share_obs_shape[:1]

    act_dim = get_shape_from_act_space(act_space)

    ppo_storage_specs = [
        StorageSpec('obs', obs_shape, torch.float32, True, 0),
        StorageSpec('share_obs', share_obs_shape, torch.float32, True, 0),
        StorageSpec('masks', (1, ), torch.uint8, True, 1),
        StorageSpec('rewards', (1, ), torch.float32, False, 0),
        StorageSpec('actions', (act_dim, ), torch.float32, False, 0),
        StorageSpec('action_log_probs', (act_dim, ), torch.float32, False, 0),
        StorageSpec('values', (1, ), torch.float32, True, 0),
        StorageSpec('rnn_states', (args.rec_n, args.hidden_size), torch.float32, True, 0),
        StorageSpec('rnn_states_critic', (args.rec_n, args.hidden_size), torch.float32, True, 0),
    ]

    if isinstance(act_space, gym.spaces.Discrete):
        ppo_storage_specs.append(StorageSpec('availalbe_actions', (act_space.n, ), torch.uint8, True, 1))

    if args.use_active_masks:
        ppo_storage_specs.append(StorageSpec('active_masks', (1, ), torch.uint8, True, 1))

    if args.use_fct_masks:
        ppo_storage_specs.append(StorageSpec('fct_masks', (1, ), torch.uint8, True, 1))

    policy_input_keys = ['share_obs', 'obs', 'available_actions', 'masks', 'rnn_states', 'rnn_states_critic']
    policy_output_keys = ['actions', 'action_log_probs', 'values', 'rnn_states', 'rnn_states_critic']
    return ppo_storage_specs, policy_input_keys, policy_output_keys
