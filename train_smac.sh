#!/bin/sh
env="StarCraft2"
algo="rmappo"
bash clean.sh

map="6h_vs_8z"
episode_length=200

seeds=(952 168 4356)
num_env_steps=20000000

for seed in ${seeds[@]};
do
    exp="SEED-RL_"${map}
    echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
    python3.8 train_smac.py --env_name ${env} \
                            --algorithm_name ${algo} \
                            --experiment_name ${exp} \
                            --group_name ${map}"_2x_r10" \
                            --map_name ${map} \
                            --seed ${seed} \
                            --n_training_threads 16 \
                            --n_eval_rollout_threads 4 \
                            --episode_length ${episode_length} \
                            --num_env_steps ${num_env_steps} \
                            --ppo_epoch 10 \
                            --use_value_active_masks \
                            --add_center_xy \
                            --use_state_agent \
                            --num_actors 16 \
                            --env_per_actor 4 \
                            --num_split 2 \
                            --eval_interval 5 \
                            --num_trainers 1 \
                            --num_servers 2 \
                            --slots_per_update 1 \
                            --server_gpu_ranks 3 \
                            --eval_interval 5 \
                            --use_eval
    bash clean.sh
done
