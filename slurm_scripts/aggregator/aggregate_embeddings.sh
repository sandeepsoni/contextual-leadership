#!/bin/bash

# Job name: Give your job a meaningful name

#SBATCH --job-name=acl_bert_embeddings_aggregate

# Partition: choose the correct machine, and below you can specify the number of nodes and tasks. 
# This is a magical mystery resolved by reading savio documentation or consulting with their help desk, e.g. https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/

#SBATCH --partition=savio3 


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

cd $PROJECTS_DIR/scripts/aggregator
module load cuda/10.2
hostname
which python

python aggregate_embeddings.py --words-file $SCRATCH_DIR/data/pre-measurement-filters/by_overall_filters$SUFFIX.keepparts --word-embeddings-dir $SCRATCH_DIR/data/aggregator

conda deactivate
#source deactivate
