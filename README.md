# Expression and Chromatin Cell cycle deconvolution


## Abstract

## Data


## Prequisites

1. CLOCCS fits

## Generate Replication Profiles

1. Fit Replication Profiles for each replicate with MNase-seq:
   - Initialize with CLOCCS fits
   - Replicate 1 and Replicate 2
   - Outputs H1, H2, N1, N2, Fr1, Fr2, B
      - Fr1, Fr2 and B are genome-wide (per chromosome)
   - `src/fit_replication_profile.py <1/2> <output_directory>`

2. Fit Combined Replication Profile
   - Fit with combined model
   - Outputs H1, H2, N1, N2, Fr, B
      - Fr and B are genome-wide (per chromosome)
   - `src/fit_combined_replication_profile.py <output_directory>`

## Deconvolve Gene Expression (combined model)

1. Deconvolve the gene expression for all genes
   - `src/deconvolve_gene_expression.py gene_name`

## Deconvolve Chromatin

2. Deconvolve the chromatin for genome-wide (combined model)
   - `src/deconvolve_chromatin.py chrom start end`

## Analysis

1. Chromatin dynamics with Gene expression 
2. Copy correction analysis
3. Origins of replication analysis
4. Transcription factor binding analysis
5. Daughter-specific gene expression dynamics





