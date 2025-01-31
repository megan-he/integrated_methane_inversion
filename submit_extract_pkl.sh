#!/bin/bash

#SBATCH -c 32
#SBATCH -t 0-6:00
#SBATCH --mem=30gb
#SBATCH -p sapphire,seas_compute,huce_ice,huce_cascade,test
#SBATCH -J extract
#SBATCH -o slurm-%j.out

module load python
mamba activate imi_env
python -u extract_pkl.py "$1"
