#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 200G
#SBATCH -p compsci

# Example run:
# sbatch -D ./slurm-logs/ --export="PYFILE=src/vit_train_cifar.py,ARGS=''" scripts/gpu_script.sh

cd /usr/xtmp/tqtran/cell-cycle

echo "batch: Starting job on $(date)"

# activate conda and  environment
. "/usr/xtmp/tqtran/miniconda3/etc/profile.d/conda.sh"

# activate environment
conda activate cell-cycle-deconvolution

python $PYFILE $ARGS

echo $(date)
echo "batch: Completed job on $(date)"
