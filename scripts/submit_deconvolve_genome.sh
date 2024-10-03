#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolved_genome_g0066_10k_10x10_2024_10_03

ARGS="${OUTDIR}"

# 1216 10k windows in the yeast genome (12 million base pairs split into 10,000 kb windows)
# The maximal sbatch array size is 1000, so split the array batches into 1000s indexed by the BATCH argument
sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="genm_1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="genm_2" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# If each job takes 1 hour, and there are 1,216 jobs. If our throughput is 32 jobs at a time.
# How many hours will it take to finish? (1,216 jobs)  / (48 simultaneous jobs)  * (2 hour per job) 
# = 38 hours = 1.58 days
