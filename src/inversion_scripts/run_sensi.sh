#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 0-48:00
#SBATCH --mem=1000G
#SBATCH -p sapphire,huce_bigmem,seas_compute,huce_cascade,huce_intel,shared
#SBATCH -J calc_sensi
#SBATCH -o slurm-calcsensi_nov.out
#SBATCH -e slurm-calcsensi_nov.err

module load python
mamba activate imi_env
printf "Calling calc_sensi_global.py\n"
python -u src/inversion_scripts/calc_sensi_global.py; wait
printf "DONE -- calc_sensi_global.py\n\n"
