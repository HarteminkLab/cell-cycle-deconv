#!/bin/bash
# Script to create jobs to create locus plots for all genes across all chromosomes

OUTDIR=output/deconvolve_gene_expression_combined_vst_2024_01_16

# There are 5774 in total, so 5774 jobs. We will split them into 1000 batch jobs (because of a limitation on the size
# of the array on the slurm computing cluster. Therefore we also need a batch argument in ARGS)

# Each batch will run 1000 genes, second argument in ARGS is the batch index that will be multiplied against the
# array index
sbatch -a 0-999%10 -D ./slurm-logs/ --job-name="dge" -p compsci --export="PYFILE=src/deconvolve_gene_expression_runner.py,ARGS=$OUTDIR 0" scripts/cpu_array_job.sh
sbatch -a 0-999%10 -D ./slurm-logs/ --job-name="dge" -p compsci --export="PYFILE=src/deconvolve_gene_expression_runner.py,ARGS=$OUTDIR 1" scripts/cpu_array_job.sh
sbatch -a 0-999%10 -D ./slurm-logs/ --job-name="dge" -p compsci --export="PYFILE=src/deconvolve_gene_expression_runner.py,ARGS=$OUTDIR 2" scripts/cpu_array_job.sh
sbatch -a 0-999%10 -D ./slurm-logs/ --job-name="dge" -p compsci --export="PYFILE=src/deconvolve_gene_expression_runner.py,ARGS=$OUTDIR 3" scripts/cpu_array_job.sh
sbatch -a 0-999%10 -D ./slurm-logs/ --job-name="dge" -p compsci --export="PYFILE=src/deconvolve_gene_expression_runner.py,ARGS=$OUTDIR 4" scripts/cpu_array_job.sh
sbatch -a 0-773%10 -D ./slurm-logs/ --job-name="dge" -p compsci --export="PYFILE=src/deconvolve_gene_expression_runner.py,ARGS=$OUTDIR 5" scripts/cpu_array_job.sh
