#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolved_copy_correction_3sites_offset01
ARGS="${OUTDIR} 1"
sbatch -a 0-3%3 -D ./slurm-logs/ --job-name="cc" -p compsci --export="PYFILE=src/deconvolve_test_windows.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
