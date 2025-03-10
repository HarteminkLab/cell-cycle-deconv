#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/prototype_pipeline_subset

# ---------- Replication deconvolution ----------------

NUM_EPOCHS=2000
# CHROM=4
# REPLICATE=1

# ARGS="replication ${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS} True"
# sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# REPLICATE=2
# ARGS="replication ${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS} True"
# sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ARGS="combined_replication ${OUTDIR}"
# sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ---------- Expression deconvolution ----------------

ARGS="deconvolve_expression ${OUTDIR}"
sbatch -D ./slurm-logs/ --job-name="rep" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ----------------------------------------------------

# ARGS="deconvolve_chromatin_staging ${OUTDIR} 0"
# sbatch -D ./slurm-logs/ --job-name="chrom" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

for i in {1..23}; do

    # Set the arguments with the current number
    ARGS="deconvolve_chromatin_staging ${OUTDIR} $i"
    
    # Submit the job
    sbatch -D ./slurm-logs/ --job-name="chrom_$i" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
done

# output/prototype_pipeline_subset/chromatin_deconvolution/test_window_set.csv