#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/prototype_pipeline_subset

# ---------- Replication deconvolution ----------------

# NUM_EPOCHS=2000
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

# ARGS="deconvolve_expression ${OUTDIR}"
# sbatch -D ./slurm-logs/ --job-name="rep" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ---------- Test Chromatin Windows -------------------

# ARGS="deconvolve_chromatin_staging ${OUTDIR} 0"
# sbatch -D ./slurm-logs/ --job-name="chrom" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

#for i in {0..28}; do

    # Set the arguments with the current number
#    ARGS="deconvolve_chromatin_staging ${OUTDIR} $i"
    
    # Submit the job
#    sbatch -D ./slurm-logs/ --job-name="chrom_$i" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
# done


# ------------- All windows -----------------------------

# ARGS="deconvolve_chromatin_full ${OUTDIR}"
# 1216 10k windows in the yeast genome (12 million base pairs split into 10,000 kb windows)
# The maximal sbatch array size is 1000, so split the array batches into 1000s indexed by the BATCH argument
# sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="chrom_b1" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
# sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="chrom_b2" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# -------------- No copy correction ---------------------

ARGS="deconvolve_chromatin_full_no_copy ${OUTDIR}"
sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="nocc_1" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="nocc_2" -p compsci --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

