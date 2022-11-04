#!/bin/bash

# to test torch-geometric on new anaconda envs on supercloud
# Slurm sbatch options
#SBATCH --job-name test_suprcloud
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a

n_agents=3

# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "test3" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--experiment_name "delete" \
--scenario_name "navigation_graph" \
--num_agents=${n_agents} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl"
