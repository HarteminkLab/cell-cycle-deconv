#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_chromatin_2023_12_15

# There are 5774 in total, so 5774 jobs. We will split them into 1000 batch jobs (because of a limitation on the size
# of the array on the slurm computing cluster. Therefore we also need a batch argument in ARGS)

# Each batch will run 1000 genes, second argument in ARGS is the batch index that will be multiplied against the
# array index
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="dchrom" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 0" scripts/cpu_array_job.sh
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="dchrom" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 1" scripts/cpu_array_job.sh
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="dchrom" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 2" scripts/cpu_array_job.sh
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="dchrom" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 3" scripts/cpu_array_job.sh
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="dchrom" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 4" scripts/cpu_array_job.sh
sbatch -a 0-773%3 -D ./slurm-logs/ --job-name="dchrom" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 5" scripts/cpu_array_job.sh

