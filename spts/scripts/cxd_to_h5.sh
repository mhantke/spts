#!/bin/sh
#SBATCH --job-name=CXDtoH5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH --mem=10000
cxd_to_h5.py "$@"
