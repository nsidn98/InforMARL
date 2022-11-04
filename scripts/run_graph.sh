#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name graph_marl
#SBATCH -a 0-5
## SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2020b

mkdir -p out_graph
# Run the script
# script to iterate through different hyperparameters
agents=(3 5 7 9 11 15)

# execute the script with different params
python -m onpolicy.scripts.train_mpe --use_valuenorm \
--use_popart --env_name "GraphMPE" --algorithm_name "rmappo" \
--experiment_name "MAPPO_${agents[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation_graph" \
--num_agents=${agents[$SLURM_ARRAY_TASK_ID]} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
&> out_graph/out_${agents[$SLURM_ARRAY_TASK_ID]}