#!/bin/bash

# to compare mappo with graph mappo both centralised and de-centralised
# compare in a simple environment of 3 agents
# Slurm sbatch options
#SBATCH --job-name marl
#SBATCH -a 0-1
## SBATCH --gres=gpu:volta:1
#SBATCH -n 40

# Loading the required module
source /etc/profile
module load anaconda/2020b

mkdir -p out_compare32
# Run the script
# script to iterate through different hyperparameters
env_names=("GraphMPE" "MPE")
scenarios=("navigation_graph" "navigation")
models=("graph_32" "vanilla_32")

# execute the script with different params
python -m onpolicy.scripts.train_mpe --use_valuenorm --use_popart \
--project_name "compare_32" \
--env_name "${env_names[$SLURM_ARRAY_TASK_ID]}" \
--algorithm_name "rmappo" \
--experiment_name "${models[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "${scenarios[$SLURM_ARRAY_TASK_ID]}" \
--num_agents=32 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
&> out_compare32/out_${models[$SLURM_ARRAY_TASK_ID]}