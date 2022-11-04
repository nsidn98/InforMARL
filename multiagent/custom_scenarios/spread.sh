#!/bin/bash

# baseline for mpe with 7 agents for maddpg
# Slurm sbatch options
#SBATCH -a 0-4
#SBATCH -o maddpg_%a.out # name the output file
#SBATCH --job-name maddpg
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2020a

n_agents=7
n_landmarks=3
algo="maddpg"
seeds=(0 1 2 3 4)
ep_length=50

# execute the script with different seeds
python -u offpolicy/scripts/train/train_mpe.py \
--env_name "MPE" \
--algorithm_name ${algo} \
--experiment_name "${algo}_${n_agents}_${ep_length}_${seeds[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation" \
--project_name "baselines_7_50" \
--num_agents=${n_agents} \
--num_landmarks ${n_landmarks} \
--seed "${seeds[$SLURM_ARRAY_TASK_ID]}" \
--n_rollout_threads 128 \
--episode_length ${ep_length} \
--actor_train_interval_step 1 \
--tau 0.005 --lr 7e-4 \
--num_env_steps 10000000 \
--batch_size 1000 --buffer_size 500000 \
--use_reward_normalization
--user_name "marl"