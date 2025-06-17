#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/draft4_run/

# ---------- Replication deconvolution ----------------

# ARGS="combined_replication ${OUTDIR}"
# sbatch -D ./slurm-logs/ --job-name="rep${REPLICATE}" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ---------- Expression deconvolution ----------------

# ARGS="deconvolve_expression ${OUTDIR}"
# sbatch -D ./slurm-logs/ --job-name="rep" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ---------- Find Gamma ---------------

# ARGS="find_gamma_chromatin ${OUTDIR}"
# sbatch -a 0-99%12 -D ./slurm-logs/ --job-name="opt_gamma" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

# ---------- Find Kappa ---------------

#ARGS="find_kappa_chromatin ${OUTDIR}"
#sbatch -a 0-99%12 -D ./slurm-logs/ --job-name="opt_kappa" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

# ---------- Find Eta ---------------

#ARGS="find_eta_chromatin ${OUTDIR}"
#sbatch -a 0-99%12 -D ./slurm-logs/ --job-name="opt_eta" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

# ---------- Find Alpha ----------------

# ARGS="find_alpha ${OUTDIR}"
# sbatch -D ./slurm-logs/ --job-name="alpha" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# -------------- Copy corrected full deconvolution ------------------

#ARGS="deconvolve_chromatin ${OUTDIR}"
#sbatch -a 0-999%16 -D ./slurm-logs/ --job-name="cc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
#sbatch -a 0-216%16 -D ./slurm-logs/ --job-name="cc_2" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# -------------- No copy correction ---------------------

#ARGS="deconvolve_chromatin_no_copy ${OUTDIR}"
#sbatch -a 0-999%16 -D ./slurm-logs/ --job-name="ncc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh
#sbatch -a 0-216%16 -D ./slurm-logs/ --job-name="ncc_2" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=1" scripts/cpu_genomicarray_job.sh

# --------------- Find all transcripts boundaries -----------------------

ARGS="call_transcripts ${OUTDIR}"
sbatch -a 1-16%8 -D ./slurm-logs/ --job-name="txb" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS,BATCH=0" scripts/cpu_genomicarray_job.sh

# --------------- Compute TPM -------------------------------------

ARGS="compute_tpms ${OUTDIR}"
sbatch -D ./slurm-logs/ --job-name="tpm" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# ------------- Test windows for copy correction -----------------------------------

# ARGS="deconvolve_chromatin ${OUTDIR} 1037"
# sbatch -D ./slurm-logs/ --job-name="cc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
# ARGS="deconvolve_chromatin ${OUTDIR} 809"
# sbatch -D ./slurm-logs/ --job-name="cc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
# ARGS="deconvolve_chromatin ${OUTDIR} 1198"
# sbatch -D ./slurm-logs/ --job-name="cc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh

# # No copy correction sample runs
# ARGS="deconvolve_chromatin_no_copy ${OUTDIR} 1037"
# sbatch -D ./slurm-logs/ --job-name="nocc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
# ARGS="deconvolve_chromatin_no_copy ${OUTDIR} 809"
# sbatch -D ./slurm-logs/ --job-name="nocc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
# ARGS="deconvolve_chromatin_no_copy ${OUTDIR} 1198"
# sbatch -D ./slurm-logs/ --job-name="nocc_1" --export="PYFILE=pipeline/run_pipeline.py,ARGS=$ARGS" scripts/cpu_job.sh
