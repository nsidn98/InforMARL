#!/bin/bash

# Slurm sbatch options
#SBATCH -a 0-7
## SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # requesting n tasks    (processes)
#SBATCH -c 10   # requesting multiple cores per task

# Loading the required module
source /etc/profile
module load anaconda/2021a

mkdir -p out_files
# Run the script
# script to iterate through different hyperparameters

algos=('DDPG' 'MADDPG')
share_acs=('True' 'False')
share_crits=('True' 'False')
runs=()
args_acs=()
args_agents=()
args_crits=()
count=1
# iterate through all combos and make a list
for i in ${!algos[@]}; do
    for j in ${!share_acs[@]}; do
        for k in ${!share_crits[@]}; do
            args_algos+=(${algos[$i]})
            args_acs+=(${share_acs[$j]})
            args_crits+=(${share_crits[$k]})
            runs+=($count)
            count=$(( $count + 1 ))
        done
    done
done

# execute the script with different params
python -m maddpg.main_temp 'navigation' 'Navigation' \
--exp_name='simpleNavigation' \
--num_agents=3 \
--num_obstacles=3 \
--collision_rew=10 \
--agent_alg=${args_algos[$SLURM_ARRAY_TASK_ID]} \
--use_shared_obs_actor=${args_acs[$SLURM_ARRAY_TASK_ID]} \
--use_shared_obs_critic=${args_crits[$SLURM_ARRAY_TASK_ID]} \
--curr_run=${runs[$SLURM_ARRAY_TASK_ID]} \
&> out_files/out_${runs[$SLURM_ARRAY_TASK_ID]}__${args_algos[$SLURM_ARRAY_TASK_ID]}__${args_acs[$SLURM_ARRAY_TASK_ID]}__${args_crits[$SLURM_ARRAY_TASK_ID]}
