#!/bin/bash

# to get baseline results for for navigation environment for 3 agents
# supports rmatd3, rmaddpg, qmix, matd3, maddpg, mqmix
# Slurm sbatch options
#SBATCH --job-name baselines
#SBATCH -a 0-5
# SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 40 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2020a

n_agents=3
# Run the script
# script to iterate through different hyperparameters
env_names=("MPE")
scenarios=("navigation")
algos=("rmatd3" "rmaddpg" "qmix" "matd3" "maddpg" "mqmix")
seeds=(0)
episode_lengths=(25)

args_algos=()
args_seeds=()
args_ep_lengths=()

# iterate through all combos and make a list
for i in ${!algos[@]}; do
    for j in ${!seeds[@]}; do
        for k in ${!episode_lengths[@]}; do
                args_algos+=(${algos[$i]})
                args_seeds+=(${seeds[$j]})
                args_ep_lengths+=(${episode_lengths[$k]})
        done
    done
done

# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "baselines" \
--env_name "MPE" \
--algorithm_name "${args_algos[$SLURM_ARRAY_TASK_ID]}" \
--seed "${args_seeds[$SLURM_ARRAY_TASK_ID]}" \
--experiment_name "${args_algos[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation" \
--num_agents=${n_agents} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 20000000 \
--use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" 