#!/bin/sh
env="Hanabi"
algo="rmappo"
pkill -9 python3.8
pkill -9 mappo

game_version="Hanabi-Full"
episode_length=100
num_agents=2

seeds=(55 37 28)
num_env_steps=10000000000

exp="mlp_critic1e-3_entropy0.015_v0belief"
ulimit -n 20480

for seed in ${seeds[@]};
do
    exp="SEED-RL_"${num_agents}"*"${game_version}
    echo "env is ${env}, game version is ${game_version}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
    python3.8 train_hanabi.py --env_name ${env} \
                                --algorithm_name ${algo} \
                                --experiment_name ${exp} \
                                --hanabi_name ${game_version} \
                                --num_agents ${num_agents} \
                                --seed ${seed} \
                                --n_training_threads 8 \
                                --n_eval_rollout_threads 32 \
                                --num_mini_batch 1 \
                                --episode_length ${episode_length} \
                                --num_env_steps ${num_env_steps} \
                                --ppo_epoch 5 \
                                --lr 7e-4 \
                                --critic_lr 1e-3 \
                                --hidden_size 512 \
                                --layer_N 2 \
                                --entropy_coef 0.015 \
                                --num_actors 4 \
                                --env_per_actor 4 \
                                --num_split 2 \
                                --eval_interval 5 \
                                --num_trainers 1 \
                                --num_servers 4 \
                                --server_gpu_ranks 3 \
                                --use_eval \
                                --use_wandb
                                # --use_eval \
    pkill -9 python3.8
    pkill -9 mappo
done
