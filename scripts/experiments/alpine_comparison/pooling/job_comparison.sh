#!/bin/bash
#SBATCH --partition=scaling
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=random_schweiger_c15_e150_q1_%a_comparison.txt
#SBATCH --array=1-1000%8
#SBATCH --qos=normal --time=0-10:00:00
#SBATCH --exclusive
julia run_comparison.jl "$SLURM_ARRAY_TASK_ID"
