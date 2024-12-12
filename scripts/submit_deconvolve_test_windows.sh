#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolved_copy_correction
ARGS="${OUTDIR} 1"
sbatch -a 0-1%1 -D ./slurm-logs/ --job-name="clb2_cc" -p compsci --export="PYFILE=src/deconvolve_test_windows.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

OUTDIR=output/deconvolved_no_copy_correction
ARGS="${OUTDIR} 0"
sbatch -a 0-1%1 -D ./slurm-logs/ --job-name="clb2_cc" -p compsci --export="PYFILE=src/deconvolve_test_windows.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
