# Expression and Chromatin Cell cycle deconvolution


## Prequisites

1. Successful runs of CLOCCS against FACS (flow cytometry) data for each replicate
2. MNase and RNA-seq data

## Example deconvolution run for a local window

1. Load the configuration for each replicate
2. Load the chromosome and span for the window
3. Deconvolve the transcription
4. Deconvolve the chromatin
5. Plot the deconvolved result
6. Plot the raw data for comparison

## The full pipeline commands

### Transcription model

1. `construct_rna_intermediate_files` - 
2. `call_transcripts` - 
3. `compute_tpms` - 
4. `deconvolve_expression_index` - 

### Replication model

5. `combined_replication` - 

### Chromatin model

#### Learn the regularization parameters

6. `find_gamma_chromatin` - 
7. `find_kappa_chromatin` - 
8. `find_eta_chromatin` - 
9. `find_alpha` - 

#### Deconvolve the chromatin

10. `deconvolve_chromatin` - 

### Create the manuscript figures

11. `figure1_chromatin_deconvolution` - 
12. `figure2_replication` - 
13. `figure3_loci` - 
14. `figure4_copy_correction` - 
15. `figure5_6_chromatin` - 
16. `figures_antisense` - 

