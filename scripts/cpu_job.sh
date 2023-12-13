#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 2G
#SBATCH -p compsci

# Example run:
# sbatch -D ./slurm-logs/ --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS='output orfname'" scripts/cpu_script.sh

cd /usr/xtmp/tqtran/deconvolution-project

echo "batch: Starting job on $(date)"

# activate conda and  environment
. "/usr/xtmp/tqtran/miniconda3/etc/profile.d/conda.sh"

# activate environment
conda activate cell-cycle-deconvolution

python $PYFILE $ARGS

echo $(date)
echo "batch: Completed job on $(date)"

