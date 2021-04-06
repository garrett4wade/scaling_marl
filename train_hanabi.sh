#!/bin/sh
env="Hanabi"
algo="rmappo"

rm -rf /dev/shm/* /tmp/*
pkill -9 python3.8

game_version="Hanabi-Full"
episode_length=50
num_agents=2

seeds=(55 37 28)
num_env_steps=10000000000

exp="mlp_critic1e-3_entropy0.015_v0belief"

for seed in ${seeds[@]};
do
    exp="SEED-RL_"${num_agents}"*"${game_version}
    echo "env is ${env}, game version is ${game_version}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
    python3.8 train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${game_version} \
                                --experiment_name ${exp} \
                                --num_agents ${num_agents} \
                                --seed ${seed} \
                                --lr 7e-4 --critic_lr 1e-3 \
                                --hidden_size 512 --layer_N 2 \
                                --entropy_coef 0.015 \
                                --n_eval_rollout_threads 32 \
                                --episode_length ${episode_length} \
                                --num_env_steps ${num_env_steps} \
                                --ppo_epoch 5 \
                                --num_actors 24 \
                                --env_per_actor 16 \
                                --num_split 2 \
                                --num_trainers 3 \
                                --num_servers 2 \
                                --slots_per_update 1 \
                                --server_gpu_ranks 3 \
                                --use_eval \
                                --use_wandb
    pkill -9 python3.8
    rm -rf /dev/shm/*
done
rm -rf /tmp/*
