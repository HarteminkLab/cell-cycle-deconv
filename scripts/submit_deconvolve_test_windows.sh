#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolved_test_windows_g01_left

ARGS="${OUTDIR} 0.1 left"
sbatch -a 0-3%3 -D ./slurm-logs/ --job-name="g1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

OUTDIR=output/deconvolved_test_windows_g02_left
ARGS="${OUTDIR} 0.2 left"
sbatch -a 0-3%3 -D ./slurm-logs/ --job-name="g1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

OUTDIR=output/deconvolved_test_windows_g04_left
ARGS="${OUTDIR} 0.4 left"
sbatch -a 0-3%3 -D ./slurm-logs/ --job-name="g1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

# ------------------------ padding on both ends -----------------------------

OUTDIR=output/deconvolved_test_windows_g01_both
ARGS="${OUTDIR} 0.1 both"
sbatch -a 0-3%3 -D ./slurm-logs/ --job-name="g1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

OUTDIR=output/deconvolved_test_windows_g02_both
ARGS="${OUTDIR} 0.2 both"
sbatch -a 0-3%3 -D ./slurm-logs/ --job-name="g1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

OUTDIR=output/deconvolved_test_windows_g04_both
ARGS="${OUTDIR} 0.4 both"
sbatch -a 0-3%3 -D ./slurm-logs/ --job-name="g1" -p compsci --export="PYFILE=src/deconvolve_genome.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
