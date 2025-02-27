#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/prototype_pipeline

NUM_EPOCHS=1000

CHROM=4
REPLICATE=1

ARGS="${OUTDIR} replication ${REPLICATE} ${CHROM} ${NUM_EPOCHS}"
sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}_${ITR}" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
