#!/bin/bash

#SBATCH --job-name="run_gaussian"
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=memory
#SBATCH --mem-per-cpu=30G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --error=gauss.%j.err

module load 2024r1
module load python
module load py-scipy
module load py-scikit-learn
module load py-numpy

srun --unbuffered python scripts/run_gaussian.py > gaussian_output.log

