#!/bin/bash

# to train informarl (the graph version; aka our method)
# with different radius of connections

# Slurm sbatch options
#SBATCH --job-name rad_comp
#SBATCH -a 0-5
#SBATCH -o rad_comp_%a.out # name the output file
#SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 40 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a

n_agents=3
logs_folder="out_compare_rad"
mkdir -p $logs_folder
# Run the script
# script to iterate through different hyperparameters
models=("rad_0.1" "rad_0.2" "rad_0.5" "rad_1" "rad_2" "rad_5")
rads=(0.1 0.2 0.5 1 2 5)
seed_max=5

for seed in `seq ${seed_max}`; do
# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "compare_radius" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "${models[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation_graph" \
--max_edge_dist "${rads[$SLURM_ARRAY_TASK_ID]}" \
--num_agents=${n_agents} \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--auto_mini_batch_size --target_mini_batch_size 128 \
&> $logs_folder/out_${models[$SLURM_ARRAY_TASK_ID]}_${seed}
done