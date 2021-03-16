#!/bin/sh
env="Hanabi"
algo="rmappo"
bash clean.sh

game_version="Hanabi-Full"
episode_length=100
num_agents=2

seeds=(58598 64579 7860)
num_env_steps=10000000000

exp="mlp_critic1e-3_entropy0.015_v0belief"
# ulimit -n 22222

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
                                --n_training_threads 128 \
                                --n_eval_rollout_threads 1 \
                                --num_mini_batch 1 \
                                --episode_length ${episode_length} \
                                --num_env_steps ${num_env_steps} \
                                --ppo_epoch 15 \
                                --lr 7e-4 \
                                --critic_lr 1e-3 \
                                --hidden_size 512 \
                                --layer_N 2 \
                                --use_eval \
                                --use_recurrent_policy \
                                --entropy_coef 0.015 \
                                --use_wandb
    bash clean.sh
done
