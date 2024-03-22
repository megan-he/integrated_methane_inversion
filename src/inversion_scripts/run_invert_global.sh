#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 0-24:00
#SBATCH --mem=5G
#SBATCH -p sapphire,huce_bigmem,seas_compute,huce_cascade,huce_intel,shared
#SBATCH -J run_invert_global
#SBATCH -o slurm-invert.out
#SBATCH -e slurm-invert.err

module load python
mamba activate imi_env
printf "Calling invert_global.py\n"
python -u src/inversion_scripts/invert_global.py; wait
printf "DONE -- invert_global.py\n\n"
