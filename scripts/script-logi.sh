#!/bin/bash

#SBATCH --job-name="run_logistic"
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu-a100-small
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=logi.%j.out
#SBATCH --error=logi.%j.err

module load 2024r1
module load cuda
module load python
module load py-scipy
module load py-scikit-learn
module load py-numpy

srun python scripts/logistic_grid_search.py
