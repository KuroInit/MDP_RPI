#!/bin/bash -x
#SBATCH --account=laionize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=develbooster
#SBATCH --time=02:00:00
#SBATCH --output=%j_0_log.out  

cd /p/home/jusers/xu17/juwels/code/MDP_CV
source /p/scratch/ccstdl/xu17/miniconda3/bin/activate base
export CUDA_VISIBLE_DEVICES=0,1,2,3

python /p/home/jusers/xu17/juwels/code/MDP_CV/scripts/new_train.py