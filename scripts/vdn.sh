#!/bin/bash

# baseline for mpe with 3 agents for vdn
# Slurm sbatch options
#SBATCH -a 0-1
#SBATCH -o vdn_%a.out # name the output file
#SBATCH --job-name vdn
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a

logs_folder="out_baselines"
mkdir -p $logs_folder

n_agents=3
n_landmarks=3
n_obstacles=3
algo="vdn"
seed_max=5
obs_type=("global" "local")
exp_names=("${algo}_global" "${algo}_local")

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
# execute the script with different seeds
python -u baselines/offpolicy/scripts/train/train_mpe.py \
--project_name "compare_3" \
--env_name "MPE" \
--algorithm_name ${algo} \
--seed "${seed}" \
--experiment_name ${exp_names[$SLURM_ARRAY_TASK_ID]} \
--scenario_name "navigation" \
--num_agents=${n_agents} \
--num_landmarks ${n_landmarks} \
--num_obstacles ${n_obstacles} \
--collision_rew 5 \
--episode_length 25 \
--lr 7e-4 \
--use_soft_update --hard_update_interval_episode 200 \
--num_env_steps 2000000 \
--batch_size 32 \
--use_reward_normalization \
--obs_type "${obs_type[$SLURM_ARRAY_TASK_ID]}" \
--user_name "marl" \
&> $logs_folder/out_${exp_names[$SLURM_ARRAY_TASK_ID]}_${seed}
done