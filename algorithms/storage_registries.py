import gym
import torch
from utils.utils import get_obs_shapes_from_spaces, get_shape_from_act_space
from collections import namedtuple


def to_numpy_type(torch_type):
    return torch.zeros((), dtype=torch_type).numpy().dtype


StorageSpec = namedtuple('StorageSpec', ['name', 'shape', 'dtype', 'bootstrap', 'init_value'])


def get_ppo_storage_specs(args, obs_space, share_obs_space, act_space):
    obs_shape, share_obs_shape = get_obs_shapes_from_spaces(obs_space, share_obs_space)
    act_dim = get_shape_from_act_space(act_space)

    ppo_storage_specs = [
        StorageSpec('obs', obs_shape, torch.float32, True, 0),
        StorageSpec('share_obs', share_obs_shape, torch.float32, True, 0),
        StorageSpec('masks', (1, ), torch.float32, True, 1),
        StorageSpec('rewards', (1, ), torch.float32, False, 0),
        StorageSpec('actions', (act_dim, ), torch.float32, False, 0),
        StorageSpec('action_log_probs', (act_dim, ), torch.float32, False, 0),
        StorageSpec('values', (1, ), torch.float32, True, 0),
        StorageSpec('rnn_states', (args.rec_n, args.hidden_size), torch.float32, False, 0),
        StorageSpec('rnn_states_critic', (args.rec_n, args.hidden_size), torch.float32, False, 0),
    ]

    if isinstance(act_space, gym.spaces.Discrete):
        ppo_storage_specs.append(StorageSpec('available_actions', (act_space.n, ), torch.float32, True, 1))

    if not (args.no_policy_active_masks and args.no_value_active_masks):
        ppo_storage_specs.append(StorageSpec('active_masks', (1, ), torch.float32, True, 1))

    if args.use_fct_masks:
        ppo_storage_specs.append(StorageSpec('fct_masks', (1, ), torch.float32, True, 1))

    policy_input_keys = ['share_obs', 'obs', 'available_actions', 'masks', 'rnn_states', 'rnn_states_critic']
    policy_output_keys = ['actions', 'action_log_probs', 'values', 'rnn_states', 'rnn_states_critic']
    return ppo_storage_specs, policy_input_keys, policy_output_keys


SUMMARY_KEYS = {
    'StarCraft2': ['elapsed_episodes', 'winning_episodes', 'episode_return', 'episode_length'],
    'Hanabi': ['elapsed_episodes', 'episode_return', 'episode_length']
}
