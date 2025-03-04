#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/prototype_pipeline_subset

NUM_EPOCHS=1000

# CHROM=4
# REPLICATE=1

# ARGS="replication ${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS} True"
# sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}_${ITR}" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# REPLICATE=2

# ARGS="replication ${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS} True"
# sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}_${ITR}" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

ARGS="combined_replication ${OUTDIR}"
sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}_${ITR}" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
