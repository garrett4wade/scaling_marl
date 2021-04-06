#!/bin/sh
rm -rf /tmp/*
pkill -9 Main_Thread & sleep 0.5
pkill -9 python3.8 & sleep 0.5
rm -rf /dev/shm/*

env="StarCraft2"
algo="rmappo"
map="MMM2"
episode_length=400

# 7860 6457 58598 168 4356
seeds=(952)
num_env_steps=25000000

for seed in ${seeds[@]};
do
    exp="SEED-RL_"${map}
    echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
    python3.8 train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --use_value_active_masks --add_center_xy --use_state_agent \
                            --n_eval_rollout_threads 4 \
                            --seed ${seed} \
                            --episode_length ${episode_length} \
                            --num_env_steps ${num_env_steps} \
                            --group_name ${map}"_6x_r5_rnn" \
                            --ppo_epoch 5 \
                            --num_actors 16 \
                            --env_per_actor 6 \
                            --num_trainers 1 \
                            --num_servers 1 \
                            --slots_per_update 1 \
                            --server_gpu_ranks 3 \
                            --use_eval
    pkill -9 Main_Thread & sleep 0.5
    pkill -9 python3.8 & sleep 0.5
    rm -rf /dev/shm/*
done
rm -rf /tmp/*
