#!/bin/bash

# Job name: Give your job a meaningful name

#SBATCH --job-name=poisson_papers_regression

# Partition: choose the correct machine, and below you can specify the number of nodes and tasks. 
# This is a magical mystery resolved by reading savio documentation or consulting with their help desk, e.g. https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/

#SBATCH --partition=savio3_bigmem 


# All the other options

#SBATCH --time=3-00:00:00

#SBATCH --account=fc_dbamman

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mail-user=sandeepsoni@berkeley.edu

#SBATCH --mail-type=all

HOME_DIR=/global/home/users/sandeepsoni
USERS_DIR=/global/scratch/users/sandeepsoni
PROJECTS_DIR=$HOME_DIR/projects/hp-modeling
SCRATCH_DIR=$USERS_DIR/projects/hp-modeling

conda activate $USERS_DIR/envs/py39/

export TRANSFORMERS_CACHE="/global/scratch/users/sandeepsoni/models/transformers" # Keep all the models in this directory
export HF_DATASETS_CACHE="/global/scratch/users/sandeepsoni/datasets/transformers" # Keep all the datasets in this directory

cd $PROJECTS_DIR/scripts/modeling
module load cuda/10.2
hostname
which python

python papers_poisson_regression.py --paper-ids-file $SCRATCH_DIR/data/raw/s2orc_acl.p5.jsonl --input-file $SCRATCH_DIR/data/cascades/paper.counts.jsonl --coefficients-file $SCRATCH_DIR/data/experiments/002/paper_coefficients/0.tsv --regularization 0.0
conda deactivate
#source deactivate
