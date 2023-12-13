#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

#CHROMS=$(seq 1 17)

#for CHROM in $CHROMS
#do
    #echo "Submitting job for chromosome $CHROM"
    #sbatch -D ./slurm-logs/ --job-name="ccplt_$CHROM" --export="PYFILE=src/gene_plotting_runner.py,ARGS=$CHROM" scripts/cpu_job.sh
#done

# Testing one job
OUTDIR=output/deconvolve_chromatin_2023_12_12
ORFNAME=YPR119W
i=0

sbatch -D ./slurm-logs/ --job-name="deconv_$i" --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR,$ORFNAME" scripts/cpu_job.sh
