#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 0-24:00
#SBATCH --mem=5G
#SBATCH -p sapphire,huce_bigmem,seas_compute,huce_cascade,huce_intel,shared
#SBATCH -J run_gridded_posterior
#SBATCH -o slurm-post.out
#SBATCH -e slurm-post.err

module load python
mamba activate imi_env
printf "Calling make_gridded_posterior_global.py\n"
python -u src/inversion_scripts/make_gridded_posterior_global.py; wait
printf "DONE -- make_gridded_posterior_global.py\n\n"