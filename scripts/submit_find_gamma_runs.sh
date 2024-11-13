#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/find_gamma_100_1k_2024_11_13

ARGS="${OUTDIR}"

# 1216 10k windows in the yeast genome (12 million base pairs split into 10,000 kb windows)
# The maximal sbatch array size is 1000, so split the array batches into 1000s indexed by the BATCH argument
sbatch -a 0-100%25 -D ./slurm-logs/ --job-name="f_gam" -p compsci --export="PYFILE=src/find_gamma_runner.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
