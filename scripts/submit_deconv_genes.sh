#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_chromatin_2023_12_13

# There are 5774 in total, so 5774 jobs. We will try 20 at a time
# They take around 20 minutes, so let's try 20 at a time
SBATCH -a 1-5774%20 -D ./slurm-logs/ -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR" scripts/cpu_array_job.sh

