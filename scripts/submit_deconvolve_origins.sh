#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_origins_g0066_11x11_2024_09_10

ARGS="${OUTDIR} 0.0066 shared"

# 1 batch of 239 origins
sbatch -a 0-239%6 -D ./slurm-logs/ --job-name="ori_1" -p compsci --export="PYFILE=src/deconvolve_origins_runner.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genearray_job.sh

