#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/draft3_run/

# ---------- Replication deconvolution ----------------

#NUM_EPOCHS=0 # No EPOCHS, will use CLOCCS fits
#CHROM=4
#REPLICATE=1

#ARGS="replication ${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS} True"
#sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

#REPLICATE=2
#ARGS="replication ${OUTDIR} ${REPLICATE} ${CHROM} ${NUM_EPOCHS} True"
#sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ARGS="combined_replication ${OUTDIR}"
# sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ---------- Expression deconvolution ----------------

# ARGS="deconvolve_expression ${OUTDIR}"
# sbatch -D ./slurm-logs/ --job-name="rep" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ---------- Find Alpha ----------------

ARGS="find_alpha ${OUTDIR}"
sbatch -D ./slurm-logs/ --job-name="alpha" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ---------- Test Chromatin Windows -------------------

# ARGS="deconvolve_chromatin_staging ${OUTDIR} 0"
# sbatch -D ./slurm-logs/ --job-name="chrom" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

#for i in {0..28}; do

    # Set the arguments with the current number
#    ARGS="deconvolve_chromatin_staging ${OUTDIR} $i"
    
    # Submit the job
#    sbatch -D ./slurm-logs/ --job-name="chrom_$i" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
# done

# ------------- All windows -----------------------------

# ARGS="deconvolve_chromatin_full ${OUTDIR}"
# 1216 10k windows in the yeast genome (12 million base pairs split into 10,000 kb windows)
# The maximal sbatch array size is 1000, so split the array batches into 1000s indexed by the BATCH argument
# sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="chrom_b1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
# sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="chrom_b2" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# -------------- No copy correction ---------------------

# ARGS="deconvolve_chromatin_full_no_copy ${OUTDIR}"
# sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="nocc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
# sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="nocc_2" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# -------------- No daughter-specificity ------------------

#ARGS="deconvolve_chromatin_no_daughter ${OUTDIR}"
#sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="nodg1_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
#sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="nodg1_2" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# -------------- Partial daughter-specificity ------------------

#ARGS="deconvolve_chromatin_partial_daughter ${OUTDIR}"
#sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="pdg1_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
#sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="pdg1_2" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# -------------- Impute 50' replicate 2 ------------------

#ARGS="deconvolve_chromatin_partial_daughter_drop_rep2_50 ${OUTDIR}"
#sbatch -a 0-999%24 -D ./slurm-logs/ --job-name="imp_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
#sbatch -a 0-216%24 -D ./slurm-logs/ --job-name="imp_2" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# ----------------------------------------------------------
