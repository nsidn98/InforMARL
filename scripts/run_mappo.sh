#!/bin/bash

# Slurm sbatch options
#SBATCH -a 0-3
## SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2021a

mkdir -p out_mappo
# Run the script
# script to iterate through different hyperparameters
agents=(7 9 11 15)

# execute the script with different params
python -m onpolicy.scripts.train_mpe --use_valuenorm \
--use_popart --env_name "MPE" --algorithm_name "rmappo" \
--experiment_name "MAPPO_${agents[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation" \
--num_agents=${agents[$SLURM_ARRAY_TASK_ID]} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
&> out_mappo/out_${agents[$SLURM_ARRAY_TASK_ID]}
