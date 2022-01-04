#!/bin/bash

# Job name: Give your job a meaningful name

#SBATCH --job-name=acl_bert_embeddings

# Partition: choose the correct machine, and below you can specify the number of nodes and tasks. 
# This is a magical mystery resolved by reading savio documentation or consulting with their help desk, e.g. https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/

#SBATCH --partition=savio2_gpu 


# All the other options

#SBATCH --time=3-00:00:00

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

export TRANSFORMERS_CACHE="/global/scratch/users/sandeepsoni/models/transformers" # Keep all the models in this directory
export HF_DATASETS_CACHE="/global/scratch/users/sandeepsoni/datasets/transformers" # Keep all the datasets in this directory

cd $PROJECTS_DIR/scripts/embeddings-learning/
module load cuda/10.2
hostname
which python
python forward-pass.py --text-file $SCRATCH_DIR/data/raw/s2orc_acl_chunks$SUFFIX.jsonl --model-checkpoint $SCRATCH_DIR/checkpoints/bert-embeddings/checkpoint-388000/ --embeddings-file $SCRATCH_DIR/data/contextual-embeddings/s2orc_acl_chunks$SUFFIX.tsv

conda deactivate
#source deactivate
