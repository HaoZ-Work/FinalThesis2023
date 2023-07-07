#!/bin/bash
#
#SBATCH --job-name=graph
#SBATCH --output=stdout/graph.txt
#SBATCH --mail-user=haritz.puerto@tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --account=ukp-researcher
#SBATCH --partition=ukp #=testing # yolo:less waiting, no promise
#SBATCH --ntasks=1 # for parallel jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
# --qos=yolo #only if partition == yolo (you need too add SBATCH like before)
#SBATCH --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180" # you can require a specific gpu (you need too add SBATCH like before)
# --constraint="gpu_model:a180" # you can require a specific gpu (you need too add SBATCH like before)
# --gres=gpumem:32g # you can require a specific gpu memory (you need too add SBATCH like before)

# a100:40gb, a6000:48gb, a180:[24gb?]
# l40:48gb,
# h100pcie:80gb
# p100:16gb, v100:16gb
# titian rtx24GB,


# configure the path to your conda env
__conda_setup="$('/ukp-storage-1/puerto/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ukp-storage-1/puerto/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/ukp-storage-1/puerto/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/ukp-storage-1/puerto/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate YOUR_CONDA_ENV
# module purge
# module load cuda/11.0 # you can change the cuda version

export NEPTUNE_API_TOKEN="YOUR API KEY TO NEPTUNE.AI IF YOU USE IT"
export NEPTUNE_PROJECT='THE NAME OF YOUR PROJECT IN NEPTUNE.AI'

nvidia-smi

python YOUR_SCRIPT.py


