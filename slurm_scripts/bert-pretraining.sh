#!/bin/bash

# Job name: Give your job a meaningful name

#SBATCH --job-name=acl_bert_embeddings

# Partition: choose the correct machine, and below you can specify the number of nodes and tasks. 
# This is a magical mystery resolved by reading savio documentation or consulting with their help desk, e.g. https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/

#SBATCH --partition=savio3_gpu 


# All the other options

#SBATCH --time=3-00:00:00

#SBATCH --account=fc_dbamman

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:V100:1

#SBATCH --qos=v100_gpu3_normal

#SBATCH --mail-user=sandeepsoni@berkeley.edu

#SBATCH --mail-type=all


PROJECTS_DIR=/global/home/users/sandeepsoni/projects/hp-modeling
SCRATCH_DIR=/global/scratch/users/sandeepsoni/projects/hp-modeling

conda activate py39
export TRANSFORMERS_CACHE="/global/scratch/users/sandeepsoni/models/transformers" # Keep all the models in this directory
export HF_DATASETS_CACHE="/global/scratch/users/sandeepsoni/datasets/transformers" # Keep all the datasets in this directory

cd $PROJECTS_DIR/scripts/embeddings-learning/
module load cuda/10.2
hostname
which python
python pretrain-mlm.py --data-file $SCRATCH_DIR/data/raw/s2orc_acl.p5.jsonl --checkpoints-dir $SCRATCH_DIR/checkpoints/bert-embeddings --num-train-epochs 10

conda deactivate
#source deactivate
