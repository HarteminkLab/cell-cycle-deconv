# CyCLOPS (Cyclic Chromatin Landscape Occupancy Profiling System)
A framework to deconvolve the chromatin and transcriptional landscape throughout the cell cycle.

## Prequisites

For our study, we generate two replicate experiments synchronized and released from alpha-factor. For each replicate, the flow cytometry, transcription state (through RNA-seq), and chromatin state (through MNase-seq) are collected.

Using the flow cytometry data, we use [CLOCCS](https://users.cs.duke.edu/~amink/software/cloccs/documentation/) to generate cell cycle parameter estimations to inform our deconvolution framework.

1. Synchronized experimental data: flow cytometry data, cellular and genomic assays (e.g. RNA-seq and MNase-seq). Replicate data recommended.
2. Successful runs of CLOCCS against flow cytometry data.

## The full deconvolution pipeline

The CyCLOPS deconvolution framework has three high-level components: (1) the transcription deconvolution model, (2) the replication deconvolution model, and (3) the chromatin deconvolution model.

For each deconvolution model, a set of intermediate files are generated from BAM into pandas high density file storage. This conversion allows for quicker reading from disk.

For each deconvolution component, the alpha parameter must be computed (the estimated delay between cytokinesis and cell wall degradation). Then, the transcription model is independent from the chromatin models. The copy correction in the chromatin deconvolution model relies on the completed replication profile estimation.

### Data preparation and setup

1. `construct_rna_intermediate_files` - Read in the RNA-seq data from BAM and generate the hdf (pandas data storage) files.
2. `construct_mnase_intermediate_files` - Read in the MNase-seq data from BAM and generate the hdf (pandas data storage) files.
3. `find_alpha` - The alpha parameter defines the estimated delay between cytokinesis and complete cell wall degradation. This value handles the fact that flow cytometry misclassifies joined mother-daughter cells with intact cell walls as a single cell with two copies of DNA. Using daughter-specific gene expression, we estimate this parameter as the optimal alpha for which daughter-specific gene expression is within the daughter-specific G1 phase.

### Transcription model

1. `call_transcripts` - Call transcript boundaries for the entire genome. This function identifies non-genic transcripts as well as identifies TSSes for genes.
2. `compute_tpms` - Compute the TPM (transcripts per million) calculation for all transcripts (genes and nongenic).
3. `deconvolve_expression_index` - Deconvolve the transcription for a gene or non-genic transcript.

### Replication model

1. `combined_replication` - Compute the replication profile for all chromosomes using the MNase data

### Chromatin model

#### Learn chromatin-specific regularization parameters

1. `find_gamma_chromatin` - Deconvolving gene expression identifies the optimal smoothing regularization term for each gene in less than a minute. However, the chromatin has millions of individual bins, so we estimate and use a single shared gamma value for the chromatin for all of the genome. For 100 random windows in the genome, find the optimal gamma value to balance smoothing and fit.
2. `find_kappa_chromatin` - Using daughter-specific genes, compute an optimal value of kappa to identify an appropriate amount of daughter-specific chromatin differences.
3. `find_eta_chromatin` - For 100 random windows, identify an optimal eta value to regularize the difference between halted cells and the recovery G1 phase.

#### Deconvolve the chromatin

10. `deconvolve_chromatin` - Deconvolve the chromatin for a specified 10kb window of the genome.

## Example deconvolution run for a local window

1. Load the configuration for each replicate
2. Load the chromosome and span for the window
3. Deconvolve the transcription
4. Deconvolve the chromatin
5. Plot the deconvolved result
6. Plot the raw data for comparison