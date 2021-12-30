#!/bin/bash

# Job name: Give your job a meaningful name

#SBATCH --job-name=gpu_demo_example

# Partition: choose the correct machine, and below you can specify the number of nodes and tasks. 
# This is a magical mystery resolved by reading savio documentation or consulting with their help desk, e.g. https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/

#SBATCH --partition=savio2_gpu 


# All the other options

#SBATCH --time=1-00:00:00

#SBATCH --account=fc_dbamman

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:1

#SBATCH --mail-user=sandeepsoni@berkeley.edu

#SBATCH --mail-type=all


HOME_DIR=/global/home/users/sandeepsoni
USERS_DIR=/global/scratch/users/sandeepsoni
PROJECTS_DIR=$HOME_DIR/projects/hp-modeling
SCRATCH_DIR=$USERS_DIR/projects/hp-modeling

conda activate $USERS_DIR/envs/py39/


cd $PROJECTS_DIR/scripts/examples
module load cuda/10.2
hostname
which python
python torch_gpu_demo.py

conda deactivate
