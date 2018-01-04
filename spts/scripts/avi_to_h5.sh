#!/bin/sh
#
#This is an example script for running SLURM jobs
#
# Set the name of the job
#SBATCH --job-name=AVItoH5
#
# Ask for 1 tasks(processes)
#SBATCH --ntasks=1
#
# Choose what resources each task will use.
# We will ask for 1 CPU per task
#SBATCH --cpus-per-task=1
#
# The partition to use.
#SBATCH -p regular
avi_to_h5.py "$@"
