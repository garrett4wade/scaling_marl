#!/bin/sh
env="StarCraft2"
map="3m"
algo="rmappo"
exp="test"
seed_max=1

pkill -9 Main_Thread
pkill -9 python3.8
pkill -9 rmappo

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python3.8 train_smac.py --env_name ${env} \
                               --algorithm_name ${algo} \
                               --experiment_name ${exp} \
                               --map_name ${map} \
                               --seed 50 \
                               --num_actors 4 \
                               --env_per_actor 4 \
                               --num_split 4 \
                               --n_training_threads 8 \
                               --num_mini_batch 1 \
                               --episode_length 60 \
                               --num_env_steps 10000000 \
                               --ppo_epoch 5 --use_value_active_masks \
                               --use_eval \
                               --add_center_xy \
                               --use_state_agent \
                               --use_recurrent_policy \
                               --use_wandb
done
