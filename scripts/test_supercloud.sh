#!/bin/bash

# to compare mappo with graph mappo both centralised and de-centralised
# compare in a simple environment of 7 agents
# Slurm sbatch options
#SBATCH --job-name test_suprcloud
#SBATCH -a 0-1
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2021b

n_agents=7
logs_folder="out"
mkdir -p $logs_folder
# Run the script
# script to iterate through different hyperparameters
env_names=("MPE" "GraphMPE")
scenarios=("navigation" "navigation_graph")
models=("vanilla_test" "graph_test")
seeds=(4)
episode_lengths=(25)

args_models=()
args_env_names=()
args_scenarios=()
args_seeds=()
args_ep_lengths=()

# iterate through all combos and make a list
for i in ${!models[@]}; do
    for j in ${!seeds[@]}; do
        for k in ${!episode_lengths[@]}; do
            args_models+=(${models[$i]})
            args_env_names+=(${env_names[$i]})
            args_scenarios+=(${scenarios[$i]})
            args_seeds+=(${seeds[$j]})
            args_ep_lengths+=(${episode_lengths[$k]})
        done
    done
done

# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "compare_7" \
--env_name "${args_env_names[$SLURM_ARRAY_TASK_ID]}" \
--algorithm_name "rmappo" \
--seed "${args_seeds[$SLURM_ARRAY_TASK_ID]}" \
--experiment_name "${args_models[$SLURM_ARRAY_TASK_ID]}_${args_ep_lengths[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "${args_scenarios[$SLURM_ARRAY_TASK_ID]}" \
--num_agents=${n_agents} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length ${args_ep_lengths[$SLURM_ARRAY_TASK_ID]} \
--num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl"
