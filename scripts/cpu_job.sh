#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem 12G
#SBATCH -p compsci

# Example run:
# sbatch -D ./slurm-logs/ --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS='output orfname'" scripts/cpu_script.sh

cd /usr/xtmp/tqtran/deconvolution-project

echo "cpu_job.sh: beginning job, date: $(date)"

# activate conda and  environment
. "/usr/project/compbio/tqtran/miniconda3/etc/profile.d/conda.sh"

# activate environment
conda activate chromatin-deconvolution

export MOSEKLM_LICENSE_FILE=/home/users/tqt3/mosek/mosek.lic

echo "bash: outputs: $ARGS"

# Run the python command with the task id (array index) as the last argument
python $PYFILE $ARGS

echo $(date)
echo "cpu_job.sh: completed job, date: $(date)"

