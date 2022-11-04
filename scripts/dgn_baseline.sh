#!/bin/bash

# baseline for graph convolutional reinforcement learning 

# Slurm sbatch options
#SBATCH --job-name dgn
#SBATCH -a 0-7
## SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 10 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b

# Run the script
# script to iterate through different hyperparameters
n_agents=(7 7 10 10 7 7 10 10)
ep_lens=(25 50 25 50 25 50 25 50)
algo_names=('dgn' 'dgn' 'dgn' 'dgn' 'dgn_atoc' 'dgn_atoc' 'dgn_atoc' 'dgn_atoc')
logs_folder="out_dgn${n_agents[$SLURM_ARRAY_TASK_ID]}"
mkdir -p $logs_folder
seed_max=5
# model_name=("dgn" "dgn_atoc")

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
# execute the script with different params
python -u -W ignore baselines/dgn/dgn_navigation/main.py \
--project_name "compare_${n_agents[$SLURM_ARRAY_TASK_ID]}" \
--env_name "MPE" \
--algorithm_name "dgn" \
--seed ${seed} \
--experiment_name ${algo_names[$SLURM_ARRAY_TASK_ID]}_${ep_lens[$SLURM_ARRAY_TASK_ID]} \
--scenario_name "navigation_dgn" \
--model_name ${algo_names[$SLURM_ARRAY_TASK_ID]} \
--num_agents ${n_agents} \
--episode_length ${ep_lens[$SLURM_ARRAY_TASK_ID]} \
--num_env_steps 5000000 \
--user_name "marl" \
&> $logs_folder/out_${algo_names[SLURM_ARRAY_TASK_ID]}_${ep_lens[$SLURM_ARRAY_TASK_ID]}_${seed}
done