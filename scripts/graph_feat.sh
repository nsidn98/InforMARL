#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name cent_obs_3
#SBATCH -a 0-3
#SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 40 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2020a

n_agents=3
logs_folder="out_graph_feat"
mkdir -p $logs_folder
# Run the script
# script to iterate through different hyperparameters
models=("graph_global_cent" "graph_global_no_cent" "graph_rel_cent" "graph_rel_no_cent")
cent_obs=("True" "False" "True" "False")
graph_feat=("global" "global" "relative" "relative")

# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "graph_feat" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--collision_rew 5 \
--experiment_name "${models[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation_graph" \
--num_agents=${n_agents} --num_obstacles 3 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs ${cent_obs[$SLURM_ARRAY_TASK_ID]} \
--graph_feat_type ${graph_feat[$SLURM_ARRAY_TASK_ID]} \
--auto_mini_batch_size --target_mini_batch_size 128 \
&> $logs_folder/out_${models[$SLURM_ARRAY_TASK_ID]}