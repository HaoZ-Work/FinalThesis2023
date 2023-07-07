#!/bin/bash
#
#SBATCH --job-name=t5-eval-csqa
#SBATCH --output=/ukp-storage-1/zhang/slurm_logs/t5-eval-csqa-%j.out
#SBATCH --mail-user=hao.zhang@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1 # for parallel jobs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a180"


source activate /mnt/beegfs/work/zhang/conda/dragon
module avail

module purge
module load cuda/11.1 # you can change the cuda version

nvidia-smi



# training
cd /ukp-storage-1/zhang/FinalThesis2023/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/beegfs/work/zhang/conda/env/lib
export WANDB_CACHE_DIR=/ukp-storage-1/zhang/wandb/cache
export WANDB_CONFIG_DIR=/ukp-storage-1/zhang/wandb/config

cd /ukp-storage-1/zhang/FinalThesis2023/
bash scripts/T5_run_eval__csqa.sh
