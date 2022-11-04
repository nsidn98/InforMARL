#!/bin/bash

# baseline for mpe with 3 agents for matd3
# Slurm sbatch options
#SBATCH -a 0-1
#SBATCH -o matd3_%a.out # name the output file
#SBATCH --job-name matd3
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a

logs_folder="out_baselines"
mkdir -p $logs_folder

n_agents=3
n_landmarks=3
n_obstacles=3
algo="matd3"
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
--n_rollout_threads 128 \
--episode_length 25 \
--tau 0.005 --lr 7e-4 \
--num_env_steps 2000000 \
--batch_size 1000 --buffer_size 500000 \
--use_reward_normalization \
--obs_type "${obs_type[$SLURM_ARRAY_TASK_ID]}" \
--user_name "marl" \
&> $logs_folder/out_${exp_names[$SLURM_ARRAY_TASK_ID]}_${seed}
done