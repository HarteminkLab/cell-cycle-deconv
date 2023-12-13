#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes


# Testing one job

OUTDIR=output/deconvolve_chromatin_2023_12_13
ORFNAME=YPR119W


# There are 5774 in total, so 5774 jobs. We will try 10 at a time

# Exmample: SBATCH -a 1-100%10 -p compsci

# Trial run of 20 jobs, with 10 concurrent
SBATCH -a 1-20 % 10 -D ./slurm-logs/ -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR" scripts/cpu_array_job.sh
