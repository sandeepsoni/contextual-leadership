#!/bin/bash

for y in $(seq 2000 2009); do
sbatch --export=ALL,HIST_WINDOW=30,REG='1000.0',REG_FILE='1000',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
sbatch --export=ALL,HIST_WINDOW=30,REG='100.0',REG_FILE='100',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
sbatch --export=ALL,HIST_WINDOW=30,REG='10.0',REG_FILE='10',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
sbatch --export=ALL,HIST_WINDOW=30,REG='1.0',REG_FILE='1',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
sbatch --export=ALL,HIST_WINDOW=30,REG='0.0',REG_FILE='0',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
sbatch --export=ALL,HIST_WINDOW=30,REG='0.1',REG_FILE='01',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
sbatch --export=ALL,HIST_WINDOW=30,REG='0.01',REG_FILE='001',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
sbatch --export=ALL,HIST_WINDOW=30,REG='0.001',REG_FILE='0001',YEAR=$y papers_poisson_regression_yearly_for_sem_innovs.sh
done
