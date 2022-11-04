#!/bin/bash

##### NOTE THIS FILE IS INCOMPLETE BEWARE #####
##### NEED TO FIX THE PARAMS TO RUN SCRIPTS ####
##### DO NOT RUN THIS ON SUPERCLOUD #####
# baseline for mpe with 7 agents for maddpg
# Slurm sbatch options
#SBATCH -a 0-4
#SBATCH -o rmatd3_%a.out # name the output file
#SBATCH --job-name rmatd3
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2020a

n_agents=3
n_landmarks=3
algo=("maddpg" "matd3" "mqmix" "rmaddpg" "rmatd3" "qmix")
scenarios=("simple_spread" "simple_reference" "simple_speaker_listener")
seeds=(0 1 2 3 4)
ep_length=25

# execute the script with different seeds
python -u offpolicy/scripts/train/train_mpe.py \
--env_name "MPE" \
--algorithm_name ${algo} \
--experiment_name "${algo}_${n_agents}" \
--scenario_name ${scenario} \
--project_name "baselines_${scenario}" \
--num_agents=${n_agents} \
--num_landmarks ${n_landmarks} \
--seed "${seeds[$SLURM_ARRAY_TASK_ID]}" \
--episode_length ${ep_length} \
--tau 0.005 --lr 7e-4 \
--num_env_steps 10000000 \
--use_reward_normalization \
--user_name "marl"