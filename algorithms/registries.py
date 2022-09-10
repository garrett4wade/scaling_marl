# import gym
import torch
from utils.utils import get_shape_from_act_space
from collections import namedtuple


def to_numpy_type(torch_type):
    return torch.zeros((), dtype=torch_type).numpy().dtype


StorageSpec = namedtuple('StorageSpec', ['name', 'shape', 'dtype', 'bootstrap', 'init_value'])


def get_ppo_storage_specs(args, obs_space, act_space):
    act_dim = get_shape_from_act_space(act_space)

    ppo_storage_specs = [
        StorageSpec('masks', (1, ), torch.uint8, True, 1),
        StorageSpec('rewards', (1, ), torch.float32, False, 0),
        StorageSpec('actions', (act_dim, ), torch.float32, False, 0),
        StorageSpec('action_log_probs', (act_dim, ), torch.float32, False, 0),
        StorageSpec('values', (3, ), torch.float32, True, 0),
        StorageSpec('rnn_states', (args.rec_n, args.hidden_size * 2), torch.float32, False, 0),
        StorageSpec('rnn_states_critic', (args.rec_n, args.hidden_size * 2), torch.float32, False, 0),
    ]

    for k, obs_shape in obs_space.items():
        if k == 'observation_self':
            ppo_storage_specs.append(StorageSpec(k, (obs_shape[-1], ), torch.float32, True, 0))
        else:
            dtype = torch.uint8 if 'mask' in k else torch.float32
            ppo_storage_specs.append(StorageSpec(k, obs_shape, dtype, True, 0))

    if args.use_fct_masks:
        ppo_storage_specs.append(StorageSpec('fct_masks', (1, ), torch.uint8, True, 1))

    policy_input_keys = [*obs_space.keys(), 'masks', 'rnn_states', 'rnn_states_critic']
    policy_output_keys = ['actions', 'action_log_probs', 'values', 'rnn_states', 'rnn_states_critic']
    return ppo_storage_specs, policy_input_keys, policy_output_keys


ENV_SUMMARY_KEYS = {
    'StarCraft2': ['elapsed_episodes', 'winning_episodes', 'episode_return', 'episode_length'],
    'Hanabi': ['elapsed_episodes', 'episode_return', 'episode_length'],
    'HideAndSeek': [
        'max_box_move_prep', 'max_box_move', 'num_box_lock_prep', 'num_box_lock', 'max_ramp_move_prep', 'max_ramp_move',
        'num_ramp_lock_prep', 'num_ramp_lock', 'episode_return_hider', 'episode_return_seeker', 'pure_hider_return', 'pure_seeker_return', 'elapsed_episodes'
    ],
}

ALGORITHM_SUMMARY_KEYS = {
    'rmappo': ['value_loss', 'policy_loss', 'dist_entropy', 'grad_norm'],
}
