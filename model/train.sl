#!/bin/bash
#SBATCH --job-name=sensitivity           # Job name
#SBATCH --output=out_%j.log  # Output log file (%j will be replaced with the job ID)
#SBATCH --error=error_%j.err # Error log file (%j will be replaced with the job ID)
#SBATCH --time=7-00:00:00                 # Maximum runtime
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a single COMSOL instance)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --mem=16GB                      # Memory per node (e.g., 32 GB)
 
# Define variables for the COMSOL run
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

python "./train_cell_03_holdout.py" 

