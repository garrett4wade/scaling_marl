#!/bin/sh
env="StarCraft2"
algo="mappo"
bash clean.sh

map="6h_vs_8z"
episode_length=400

# 168 4356 
seeds=(952)
num_env_steps=15000000

for seed in ${seeds[@]};
do
    exp="SEED-RL_"${map}
    echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
    python3.8 train_smac.py --env_name ${env} \
                            --algorithm_name ${algo} \
                            --experiment_name ${exp} \
                            --map_name ${map} \
                            --seed ${seed} \
                            --n_training_threads 8 \
                            --num_mini_batch 1 \
                            --n_eval_rollout_threads 4 \
                            --episode_length ${episode_length} \
                            --num_env_steps ${num_env_steps} \
                            --ppo_epoch 5 \
                            --use_value_active_masks \
                            --add_center_xy \
                            --use_state_agent \
                            --num_actors 4 \
                            --env_per_actor 2 \
                            --num_split 2 \
                            --eval_interval 5 \
                            --use_eval \
                            --use_wandb
    bash clean.sh
done
