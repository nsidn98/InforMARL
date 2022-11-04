#!/bin/bash

# to train informarl (the graph version; aka our method)

# Slurm sbatch options
#SBATCH --job-name informarl7
#SBATCH -a 0-3
## SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 40 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a

logs_folder="out_informarl7"
mkdir -p $logs_folder
# Run the script
# seed_max=4
n_agents=7
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
seeds=(2 3 4 5)

# for seed in `seq ${seed_max}`;
# do
# seed=`expr ${seed} + 1`
echo "seed: ${seeds[$SLURM_ARRAY_TASK_ID]}"
# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "compare_${n_agents}" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
--experiment_name "informarl_25" \
--scenario_name "navigation_graph" \
--num_agents=${n_agents} \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 5000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128 \
&> $logs_folder/out_25_${seeds[$SLURM_ARRAY_TASK_ID]}
done