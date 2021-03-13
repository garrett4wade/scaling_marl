#!/bin/sh
env="StarCraft2"
algo="rmappo"
bash clean.sh

map="3m"
episode_length=70

seeds=(64579 7860)
num_env_steps=10000000

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
                            --episode_length ${episode_length} \
                            --num_env_steps ${num_env_steps} \
                            --ppo_epoch 5 --use_value_active_masks \
                            --use_eval \
                            --add_center_xy \
                            --use_state_agent \
                            --use_recurrent_policy \
                            --num_actors 4 \
                            --env_per_actor 4 \
                            --num_split 2 \
                            --use_wandb
    bash clean.sh
done
