#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes


# This is a temporary script to deconvolve the 1007 genes that failed to complete
# because of an error in creating the named plots, the genes without names (only systematic names)
# failed early
#
# This is to resolve the 2024-02-20/22 runs. This script will no longer be needed after 2/27/24

OUTDIR=output/deconvolve_rep1_g006_2024_02_20
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="dec1_1" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=0" scripts/cpu_genearray_job.sh
sbatch -a 0-7%6 -D ./slurm-logs/ --job-name="dec1_2" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 1 0.006,BATCH=1" scripts/cpu_genearray_job.sh


OUTDIR=output/deconvolve_rep2_g006_2024_02_22
sbatch -a 0-999%6 -D ./slurm-logs/ --job-name="dec1_1" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 2 0.006,BATCH=0" scripts/cpu_genearray_job.sh
sbatch -a 0-7%6 -D ./slurm-logs/ --job-name="dec1_2" -p compsci --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR 2 0.006,BATCH=1" scripts/cpu_genearray_job.sh
