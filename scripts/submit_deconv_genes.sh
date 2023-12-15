#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_chromatin_2023_12_13

# There are 5774 in total, so 5774 jobs

# Max Array size is 1001, so let's do 6 super-batches
# sbatch -a 0-1000%10 -D ./slurm-logs/ -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR" scripts/cpu_array_job.sh

# TODO:
# Current constraint to work around, max array size is 1001, meaning we can't specify an array from 1001-2000,
# So we'll need to add another argument to the python runner to multiply the array index to the correct orf...
#

