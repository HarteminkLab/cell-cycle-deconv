#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_combined_delta_2024_05_27

# There are 5774 in total, so 5774 jobs. We will split them into 1000 batch jobs (because of a limitation on the size
# of the array on the slurm computing cluster. Therefore we also need a batch argument in ARGS)

# Each batch will run 1000 genes, second argument in ARGS is the batch index that will be multiplied against the
# array index
sbatch -a 0-999%10 -D ./slurm-logs/ --job-name="delt_1" -p compsci --export="PYFILE=src/deconvolve_combined_runner.py,ARGS=$OUTDIR,BATCH=0" scripts/cpu_genearray_job.sh
sbatch -a 0-999%10 -D ./slurm-logs/ --job-name="delt_2" -p compsci --export="PYFILE=src/deconvolve_combined_runner.py,ARGS=$OUTDIR,BATCH=1" scripts/cpu_genearray_job.sh
