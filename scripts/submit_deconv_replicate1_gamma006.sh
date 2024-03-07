#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_rep1_g006_2024_02_20

# There are 5774 in total, so 5774 jobs. We will split them into 1000 batch jobs (because of a limitation on the size
# of the array on the slurm computing cluster. Therefore we also need a batch argument in ARGS)

# Each batch will run 1000 genes, second argument in ARGS is the batch index that will be multiplied against the
# array index
sbatch -a 0-999%3 -D ./slurm-logs/ --job-name="dec1_1" -p compsci --export="PYFILE=src/deconvolve_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=0" scripts/cpu_genearray_job.sh
sbatch -a 0-999%3 -D ./slurm-logs/ --job-name="dec1_2" -p compsci --export="PYFILE=src/deconvolve_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=1" scripts/cpu_genearray_job.sh
sbatch -a 0-999%3 -D ./slurm-logs/ --job-name="dec1_3" -p compsci --export="PYFILE=src/deconvolve_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=2" scripts/cpu_genearray_job.sh
sbatch -a 0-999%3 -D ./slurm-logs/ --job-name="dec1_4" -p compsci --export="PYFILE=src/deconvolve_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=3" scripts/cpu_genearray_job.sh
sbatch -a 0-999%3 -D ./slurm-logs/ --job-name="dec1_5" -p compsci --export="PYFILE=src/deconvolve_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=4" scripts/cpu_genearray_job.sh
sbatch -a 0-773%3 -D ./slurm-logs/ --job-name="dec1_6" -p compsci --export="PYFILE=src/deconvolve_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=5" scripts/cpu_genearray_job.sh

