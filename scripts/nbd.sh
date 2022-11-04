#!/bin/bash

# to train with nbd obs for rmappo
# local and nbd obs are transferrable to more agents whereas global obs is not

# Slurm sbatch options
#SBATCH --job-name rmappo
#SBATCH -a 0-5
#SBATCH -o rmappo_%a.out # name the output file
## SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 40 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a

n_agents=10
logs_folder="out_nbd${n_agents}"
mkdir -p $logs_folder
algo="rmappo"
# Run the script
models=("${algo}_${n_agents}_nbd_5_25" "${algo}_${n_agents}_nbd_10_25" "${algo}_${n_agents}_global_25" "${algo}_${n_agents}_nbd_5_50" "${algo}_${n_agents}_nbd_10_50" "${algo}_${n_agents}_global_50")
obs_type=("nbd" "nbd" "global" "nbd" "nbd" "global")
num_nbds=(5 10 20 5 10 20)
ep_lens=(25 25 25 50 50 50)
seed_max=5

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "compare_${n_agents}" \
--env_name "MPE" \
--algorithm_name ${algo} \
--seed ${seed} \
--experiment_name "${models[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "navigation" \
--num_agents=${n_agents}  \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length ${ep_lens[$SLURM_ARRAY_TASK_ID]} \
--num_env_steps 5000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--obs_type "${obs_type[$SLURM_ARRAY_TASK_ID]}" \
--num_nbd_entities ${num_nbds[$SLURM_ARRAY_TASK_ID]} \
--auto_mini_batch_size --target_mini_batch_size 128 \
&> $logs_folder/out_${models[$SLURM_ARRAY_TASK_ID]}_${seed}
done