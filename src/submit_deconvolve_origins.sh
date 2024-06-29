#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_origins_g0066_2024_06_28

ARGS="${OUTDIR} 0.0066 shared 1"

# 1 batch of 798 origins
sbatch -a 0-798%6 -D ./slurm-logs/ --job-name="ori_1" -p compsci --export="PYFILE=src/deconvolve_origins_runner.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genearray_job.sh
