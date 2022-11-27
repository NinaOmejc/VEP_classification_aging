#!/bin/bash
#SBATCH --job-name=erp_v6
#SBATCH --time=2-00:00:00
#SBATCH --array=1-27
#SBATCH --output=./erp_classification_study/results/slurm/v6/slurm_output_%A_%a.out

cd ./erp_classification_study/

singularity exec ./src/erp_class_study_container.sif python3 ./src/classify_main.py ${SLURM_ARRAY_TASK_ID}

