#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_replication

REPLICATE=1
NUM_EPOCHS=100

#for CHROM in {1..16}; do
#	ARGS="${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS}"#
#	sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" -p compsci --export="PYFILE=src/SingleReplicateDeconvolutionRunner.py,ARGS=$ARGS" scripts/cpu_job.sh
#done

CHROM=1
sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" -p compsci --export="PYFILE=src/SingleReplicateDeconvolutionRunner.py,ARGS=$ARGS" scripts/cpu_job.sh
