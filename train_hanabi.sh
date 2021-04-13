#!/bin/sh
rm -rf /dev/shm/* /tmp/*
pkill -9 python3.8 & sleep 0.5

env="Hanabi"
algo="mappo"
game_version="Hanabi-Full"

episode_length=10
num_agents=2
replay=15

seeds=(55 37 28)
num_env_steps=10000000000  # 10B

for seed in ${seeds[@]};
do
    exp="CircularBuffer_"${num_agents}"*"${game_version}
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
                                --ppo_epoch ${replay} \
                                --num_actors 12 \
                                --env_per_actor 60 \
                                --num_split 1 \
                                --num_trainers 3 \
                                --num_servers 4 \
                                --slots_per_update 1 \
                                --server_gpu_ranks 3 \
                                --use_eval \
                                --use_wandb
    pkill -9 python3.8 & sleep 0.5
    rm -rf /dev/shm/*
done
rm -rf /tmp/*
