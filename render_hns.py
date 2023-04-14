#!/usr/bin/env python
import time
import glfw
import numpy as np
from operator import itemgetter
from mujoco_py import const, MjViewer
from mujoco_worldgen.util.types import store_args
import sys
import argparse
from os.path import abspath, dirname, join
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from envs.hns.viewer.policy_viewer import PolicyViewer
from gym.spaces import Tuple
import torch


def listdict2dictnp(lis, keepdims=False):
    '''
        Convert a list of dicts of numpy arrays to a dict of numpy arrays.
        If keepdims is False the new outer dimension in each dict element will be
            the length of the list
        If keepdims is True, then the new outdimension in each dict will be the sum of the
            outer dimensions of each item in the list
    '''
    if keepdims:
        return {k: np.concatenate([d[k] for d in lis]) for k in lis[0]}
    else:
        return {k: np.array([d[k] for d in lis]) for k in lis[0]}


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='hns', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario_name', type=str,
                        default='quadrant', help="Which scenario to run on")
    parser.add_argument('--floor_size', type=float,
                        default=6.0, help="size of floor")
    parser.add_argument('--grid_size', type=int,
                        default=30, help="size of floor")
    parser.add_argument('--door_size', type=int,
                        default=2, help="size of floor")
    parser.add_argument('--prep_fraction', type=float, default=0.4)
    parser.add_argument('--p_door_dropout',type=float, default=1.0)

    # transfer task
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument('--num_boxes', type=int,
                        default=4, help="number of boxes")
    parser.add_argument("--task_type", type=str, default='all')
    parser.add_argument("--objective_placement", type=str, default='center')

    # hide and seek task
    parser.add_argument("--num_seekers", type=int,
                        default=1, help="number of seekers")
    parser.add_argument("--num_hiders", type=int,
                        default=1, help="number of hiders")
    parser.add_argument("--num_ramps", type=int,
                        default=1, help="number of ramps")
    parser.add_argument("--num_food", type=int,
                        default=0, help="number of food")
    parser.add_argument("--max_seekers", type=int,
                        default=2, help="number of seekers")

    parser.add_argument("--training_role", type=int,
                        default=0, help="number of food")
    
    parser.add_argument("--use_partial", action='store_true', default=False)

    parser.add_argument("--use_mix", action='store_true', default=False)
    parser.add_argument("--num_seekers_max", type=int,
                        default=2, help="max number of seekers")
    
    parser.add_argument("--quadrant_game_ramp_uniform_placement", action='store_true', default=False)
    parser.add_argument("--quadrant_game_hider_uniform_placement", action='store_true', default=False)
    parser.add_argument("--grab_out_of_vision", action='store_true', default=False)
    
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    '''
    examine.py is used to display environments and run policies.

    For an example environment jsonnet, see
        mujoco-worldgen/examples/example_env_examine.jsonnet
    You can find saved policies and the in the 'examples' together with the environment they were
    trained in and the hyperparameters used. The naming used is 'examples/<env_name>.jsonnet' for
    the environment jsonnet file and 'examples/<env_name>.npz' for the policy weights file.
    Example uses:
        bin/examine.py hide_and_seek
        bin/examine.py mae_envs/envs/base.py
        bin/examine.py base n_boxes=6 n_ramps=2 n_agents=3
        bin/examine.py my_env_jsonnet.jsonnet
        bin/examine.py my_env_jsonnet.jsonnet my_policy.npz
        bin/examine.py hide_and_seek my_policy.npz n_hiders=3 n_seekers=2 n_boxes=8 n_ramps=1
        bin/examine.py examples/hide_and_seek_quadrant.jsonnet examples/hide_and_seek_quadrant.npz
    '''
    #names, kwargs = parse_arguments(argv)
    args = parse_args(args)
    kwargs={'args': args}

    env_name = "hide_and_seek"
    num_hiders = 2
    num_seekers = 1
    num_agents = num_hiders + num_seekers
    core_dir = abspath(join(dirname(__file__)))
    envs_dir = '/home/yuchao/scaling_marl/envs/hns/envs/'  # where hide_and_seek.py is.
    xmls_dir = 'xmls'

    env, args_remaining_env = load_env(env_name, core_dir=core_dir,
                                        envs_dir=envs_dir, xmls_dir=xmls_dir,
                                        return_args_remaining=True, **kwargs)
    args_remaining_env = {}
    
    if isinstance(env.action_space, Tuple):
        env = JoinMultiAgentActions(env)
    if env is None:
        raise Exception(f'Could not find environment based on pattern {env_name}')
    
    env.reset()  # generate action and observation spaces
    
    model_dir = "/home/yuchao/scaling_marl/models/HideAndSeek/rmappo/unif_reuse4_icml/policy_0/"
    policies = [torch.load(model_dir + 'model_200.pt', map_location=torch.device('cuda:0'))]

    args_remaining_policy = args_remaining_env
    
    if env is not None and policies is not None:
        args_to_pass, args_remaining_viewer = extract_matching_arguments(PolicyViewer, kwargs)
        args_remaining = set(args_remaining_env)
        args_remaining = args_remaining.intersection(set(args_remaining_policy))
        args_remaining = args_remaining.intersection(set(args_remaining_viewer))
        assert len(args_remaining) == 0, (
            f"There left unused arguments: {args_remaining}. There shouldn't be any.")
        viewer = PolicyViewer(env, policies, model_dir, **args_to_pass)
        viewer.run()

if __name__ == '__main__':
    main(sys.argv[1:])