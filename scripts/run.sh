#!/bin/bash

# Slurm sbatch options
#SBATCH -a 0-15
## SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2021a

mkdir -p out_files
# Run the script
# script to iterate through different hyperparameters
agents=(3 5 7 10)
obst=(3 5 7 10)

args_agents=()
args_obst=()
runs=()
count=1
# iterate through all combos and make a list
for i in ${!agents[@]}; do
    for j in ${!obst[@]}; do
        args_agents+=(${agents[$i]})
        args_obst+=(${obst[$j]})
        runs+=($count)
        count=$(( $count + 1 ))
    done
done
# echo "Agents: ${args_agents[@]}"
# echo "Obst: ${args_obst[@]}"
# echo "Collab: ${args_col[@]}"
# echo "Runs: ${runs[@]}"

# execute the script with different params
python -m maddpg.main 'navigation' 'Navigation' \
--exp_name='simpleNavigation' \
--num_agents=${args_agents[$SLURM_ARRAY_TASK_ID]} \
--num_obstacles=${args_obst[$SLURM_ARRAY_TASK_ID]} \
--curr_run=${runs[$SLURM_ARRAY_TASK_ID]} \
&> out_files/out_${args_agents[$SLURM_ARRAY_TASK_ID]}__${args_obst[$SLURM_ARRAY_TASK_ID]}
