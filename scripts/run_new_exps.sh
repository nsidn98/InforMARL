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
agents=()
scenario_names=('navigation_graph', 'navigation')
env_names=('GraphMPE', 'MPE')
exp_names=('graph', 'large_scale')

# execute the script with different params
python -m onpolicy.scripts.train_mpe --use_valuenorm \
--use_popart --algorithm_name "rmappo" \
--env_name ${env_names[$SLURM_ARRAY_TASK_ID]}
--experiment_name ${exp_names[$SLURM_ARRAY_TASK_ID]} \
--scenario_name ${scenario_names[$SLURM_ARRAY_TASK_ID]} \
--num_agents=${agents[$SLURM_ARRAY_TASK_ID]} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
&> out_mappo/out_${exp_names[$SLURM_ARRAY_TASK_ID]}