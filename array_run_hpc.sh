#!/bin/bash

#SBATCH --job-name=ssld_area2
#SBATCH --mail-type=END
#SBATCH --mail-user=yongxu.zhang@yale.edu
#SBATCH --output=./save/gpu_job_array_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --qos=qos_saxena
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --array=0               # Array range

pwd; hostname; date
echo This is task $SLURM_ARRAY_TASK_ID

# module load CUDA
# module load cuDNN
# using your anaconda environment
module load miniconda
conda activate BSRNN

python array_area2.py --config array_config.yaml --fold $SLURM_ARRAY_TASK_ID

conda deactivate

# --gpus=a100:1