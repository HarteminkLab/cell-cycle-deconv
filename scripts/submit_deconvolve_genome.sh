#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_sharedg1_g0066_TSS_11x11_x2_2024_09_18

ARGS="${OUTDIR}"

# There are 5774 in total, so 5774 jobs. We will split them into 1000 batch jobs (because of a limitation on the size
# of the array on the slurm computing cluster. Therefore we also need a batch argument in ARGS)
sbatch -a 0-1%6 -D ./slurm-logs/ --job-name="TSS_1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

# Each batch will run 1000 genes, second argument in ARGS is the batch index that will be multiplied against the
# array index
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="TSS_1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
sbatch -a 0-216%6 -D ./slurm-logs/ --job-name="TSS_2" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh
