#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

# OUTDIR=output/deconvolve_replication
NUM_EPOCHS=10000

# REPLICATE=1
# for CHROM in {1..16}; do
# 	ARGS="${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS}"
# 	sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" -p compsci --export="PYFILE=src/SingleReplicateDeconvolutionRunner.py,ARGS=$ARGS" scripts/cpu_job.sh
# done

# REPLICATE=2

# for CHROM in {1..16}; do
# 	ARGS="${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS}"
# 	sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" -p compsci --export="PYFILE=src/SingleReplicateDeconvolutionRunner.py,ARGS=$ARGS" scripts/cpu_job.sh
# done


CHROM=4

for ITR in {1..8}; do
    OUTDIR="output/combined_deconvolve_replication_${ITR}"
	ARGS="${OUTDIR} ${CHROM} ${NUM_EPOCHS}"
	sbatch -D ./slurm-logs/ --job-name="chr${CHROM}_${ITR}" -p compsci --export="PYFILE=src/CombinedReplicateDeconvolutionRunner.py,ARGS=$ARGS" scripts/cpu_job.sh
done

