#!/bin/bash

# baseline for graph policy gradient
# NOTE: don't change any of the hyperparams used by original paper
# Slurm sbatch options
#SBATCH --job-name gpg
#SBATCH -a 0-7
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b

n_agents=(7 7 10 10 7 7 10 10)
ep_lens=(25 50 25 50 25 50 25 50)
# Run the script
# script to iterate through different hyperparameters
logs_folder="out_gpg${n_agents[$SLURM_ARRAY_TASK_ID]}"
mkdir -p $logs_folder
exp_names=("gpg_dynamic" "gpg_dynamic" "gpg_dynamic" "gpg_dynamic" "gpg_static" "gpg_static" "gpg_static" "gpg_static")
graph_type=("dynamic" "dynamic" "dynamic" "dynamic" "static" "static" "static" "static")
seed_max=5

# execute the script with different params
for seed in `seq ${seed_max}`; do
python -u -W ignore baselines/gpg/rl_navigation/main.py \
--project_name "compare_${n_agents[$SLURM_ARRAY_TASK_ID]}" \
--env_name "MPE" \
--algorithm_name "gpg" \
--seed ${seed} \
--graph_type "${graph_type[$SLURM_ARRAY_TASK_ID]}" \
--experiment_name "${exp_names[$SLURM_ARRAY_TASK_ID]}_${ep_lens[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation_gpg" \
--num_agents ${n_agents} \
--episode_length ${ep_lens[$SLURM_ARRAY_TASK_ID]} \
--num_env_steps 5000000 \
--user_name "marl" \
&> $logs_folder/out_${exp_names[$SLURM_ARRAY_TASK_ID]}_${ep_lens[$SLURM_ARRAY_TASK_ID]}_${seed}
done
