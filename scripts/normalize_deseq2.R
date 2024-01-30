
# Testing whether normalizing is improved using DESeq2 versus TPM
# Deconvolution previously used microarray data, thus error
# was in a different form than RNA-seq, this script converts read counts
# to normalized reads that may better suit the deconvolution algorithm.
# Unclear at the moment if this is true.

library(DESeq2)

setwd('~/Research/deconvolution-python/')

read_count_to_vst <- function(read_path, save_path)  {
  # Load the gene expression count data
  counts <- read.csv(read_path, row.names = 1)
  
  sampleInfo <- data.frame(row.names = colnames(counts), condition = factor(rep("condition1", ncol(counts))))
  dds <- DESeqDataSetFromMatrix(countData = counts, colData = sampleInfo, design = ~1)
  dds <- DESeq(dds)
  
  # Normalize using the variance stabilizing transformation
  vst_data <- vst(dds, blind=FALSE)
  vst_counts <- assay(vst_data)
  write.csv(as.data.frame(vst_counts), file=save_path)
}

# For replicate 1
read_path <- "datasets/yl_cell_cycle/replicate1_gene_expression_counts.csv"
save_path <- "datasets/yl_cell_cycle/replicate1_deseq2_vst_counts.csv"
read_count_to_vst(read_path, save_path)

# For replicate 2
read_path <- "datasets/yl_cell_cycle/replicate2_gene_expression_counts.csv"
save_path <- "datasets/yl_cell_cycle/replicate2_deseq2_vst_counts.csv"
read_count_to_vst(read_path, save_path)
