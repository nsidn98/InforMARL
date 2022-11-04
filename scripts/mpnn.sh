#!/bin/bash

# baselines with mpnn

# Slurm sbatch options
#SBATCH --job-name mpnn
#SBATCH -a 0-3
#SBATCH -o mpnn_%a.out # name the output file
## SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a

algo="mpnn"
n_agents=(7 7 10 10)
ep_lens=(25 50 25 50)
# obs_types=("local" "global" "local" "global")
seed_max=5
logs_folder="out_mpnn${n_agents[$SLURM_ARRAY_TASK_ID]}"
mkdir -p $logs_folder

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
echo "log folder: ${logs_folder}/out_${algo}_${ep_lens[$SLURM_ARRAY_TASK_ID]}_${seed}"
python -u -W ignore baselines/mpnn/nav/main.py \
--scenario_name='navigation' \
--entity-mp "True" \
--project_name "compare_${n_agents}" \
--num_agents ${n_agents} \
--episode_length ${ep_lens[$SLURM_ARRAY_TASK_ID]} \
--env_name "MPE" \
--algorithm_name ${algo} \
--experiment_name "mpnn_${ep_lens[$SLURM_ARRAY_TASK_ID]}" \
--seed ${seed} \
--obs_type "global" \
--n_rollout_threads=128 \
--num_agents ${n_agents[$SLURM_ARRAY_TASK_ID]} \
--num_env_steps 5000000 \
--user_name "marl" \
&> $logs_folder/out_${algo}_${ep_lens[$SLURM_ARRAY_TASK_ID]}_${seed}
done