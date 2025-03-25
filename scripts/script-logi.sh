#!/bin/bash

#SBATCH --job-name="run_logistic"
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=memory
#SBATCH --mem-per-cpu=15G
#SBATCH --account=education-eemcs-msc-cs

module load 2024r1
module load python
module load py-scipy
module load py-scikit-learn
module load py-numpy

srun python scripts/run_logistic.py > logistic.log

