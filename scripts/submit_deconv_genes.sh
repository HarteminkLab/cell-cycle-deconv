#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes


# Testing one job

OUTDIR=output/deconvolve_chromatin_2023_12_13
ORFNAME=YPR119W

sbatch -D ./slurm-logs/ --job-name="deconv_$i" --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR $ORFNAME" scripts/cpu_job.sh



# Loop through the ORF names and create a slurm job to deconvolve each of these genes
#i=1
#while IFS= read -r line
#do
#    if [ "$line" != "orf_name" ]; then

#        ORFNAME=$line
#        sbatch -D ./slurm-logs/ --job-name="deconv_$i" --export="PYFILE=src/deconvolve_chromatin_runner.py,ARGS=$OUTDIR $ORFNAME" scripts/cpu_job.sh
#       i=$(($i+1))

#    fi

# We are reading in the first column of the csv using cut ("-f1") which defines the orf name
#done < <(cut -d ',' -f1 data/reference_data/geneset_nondub_w_prom_genebodies.csv)

