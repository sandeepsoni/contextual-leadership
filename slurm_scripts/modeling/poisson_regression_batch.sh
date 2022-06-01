#!/bin/bash

sbatch --export=ALL,HIST_WINDOW=3,REG='1000.0',REG_FILE='1000' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='100.0',REG_FILE='100' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='10.0',REG_FILE='10' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='1.0',REG_FILE='1' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='0.0',REG_FILE='0' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='0.1',REG_FILE='0.1' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='0.1',REG_FILE='01' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='0.01',REG_FILE='001' papers_poisson_regression.sh
sbatch --export=ALL,HIST_WINDOW=3,REG='0.001',REG_FILE='0001' papers_poisson_regression.sh

