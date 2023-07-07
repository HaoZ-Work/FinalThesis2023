#!/bin/bash
#
#SBATCH --job-name=t5-fin-tuning-csqa
#SBATCH --output=/ukp-storage-1/zhang/slurm_logs/t5-fin-tuning-csqa-%j.out
#SBATCH --mail-user=hao.zhang@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1 # for parallel jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a100|gpu_model:a6000|gpu_model:a180"


source activate /mnt/beegfs/work/zhang/conda/dragon

module purge
module load cuda/11.0 # you can change the cuda version

nvidia-smi



# training
cd /ukp-storage-1/zhang/FinalThesis2023/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/beegfs/work/zhang/conda/env/lib
export WANDB_CACHE_DIR=/ukp-storage-1/zhang/wandb/cache
export WANDB_CONFIG_DIR=/ukp-storage-1/zhang/wandb/config

python3 -u T5_finetuning/run_wandb_agent.py
