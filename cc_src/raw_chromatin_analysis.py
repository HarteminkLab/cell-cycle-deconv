
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from cc_src.mnase_reads import load_mnase_reads
from cc_src.reference_data import load_analysis_genes
from cc_src.chromatin_metrics import yl_rep2_len_spans


class ChromatinDataAnalysis:
	"""Class to analyze raw chromatin data, for
	plotting and comparison of gene windows"""

	def __init__(self):

		# load geneset
		self.genes = load_analysis_genes()
		self.window = 3000
		self.n = len(self.genes)

	def set_chrom_replicate(self, chromosome, replicate):
		"""Load the MNase-seq data for each chromosome for each gene and bin
		May be a lot of data to hold in memory, so create intermediate files
		for each chromosome"""

		self.chromosome = chromosome
		self.replicate = replicate

		self.chr_genes = self.genes[self.genes.chr == chromosome]
		self.chr_reads = load_mnase_reads(chromosome, replicate)
		self.normalizion_scaling = pd.read_csv('datasets/computed_mnase/rep1_len_counts_fix.csv')
		self.gene_plus_ones = pd.read_csv('datasets/computed_mnase/rep1_plus_ones.csv').set_index('orf_name')
		self.samples = self.chr_reads['sample'].unique()
		self.m = len(self.samples)

	def create_gene_mnase_histogram(self):

		# Initialize data structures
		self.chr_gene_read_counts = np.zeros((self.n, self.m, self.window))
		self.chr_gene_nuc_read_counts = np.zeros((self.n, self.m, self.window))
		self.chr_gene_small_read_counts = np.zeros((self.n, self.m, self.window))

		# Loop through all chromosome genes
		for orf_name, gene in self.chr_genes:
			count, nuc_count, sm_count = self.load_gene_read_counts(gene)

	def load_gene_read_counts(self, gene):
		"""Load the gene reads as a 2D histogram"""

		small_span, mid_span, nuc_span = yl_rep2_len_spans()

		from cc_src.mnase_reads import filter_reads

		chr_reads = self.chr_reads
		plus_one = self.gene_plus_ones.loc[gene.name]['+1']

		window_2 = self.window//2
		gene_span = plus_one-window_2, plus_one+window_2
		len_span = 0, 250

		# Load the reads for the gene
		gene_sample_counts = np.zeros((self.m, (len_span[1]-len_span[0]), self.window))

		# For each sample, store into a numpy array the counts at each genomic position
		for i in range(self.m):
			sample = self.samples[i]
			gene_sample_reads = filter_reads(chr_reads, gene_span[0], gene_span[1], sample=sample)
			gene_reads_hist2d = bin_reads(gene_sample_reads, gene_span, len_span)
			gene_sample_counts[i] = gene_reads_hist2d

		# Flip if strand is crick
		if gene.strand == '-':
			gene_sample_counts = gene_sample_counts[:, :, ::-1]

		# Normalize the data by length
		
		return gene_sample_counts


def bin_reads(reads, span, len_span):
	x_bins = np.arange(span[0], span[1]+1)
	y_bins = np.arange(len_span[0], len_span[1]+1)
	counts, _, _ = np.histogram2d(reads['mid'], 
		reads['length'], bins=[x_bins, y_bins])
	return counts.T


def plot_im(ax, gene_reads_hist2d, vmax=1):
	ax.imshow(gene_reads_hist2d, origin='lower', cmap='magma_r', vmax=vmax, aspect='auto')
	ax.set_xticks([])
	ax.set_yticks([])

	
def plot_gene_series(analysis, gene_sample_counts):
	fig, axs = plt.subplots(analysis.m, 1, figsize=(3, 6))
	plt.subplots_adjust(hspace=0.0)

	for i in range(len(axs)):
		ax = axs[i]
		plot_im(ax, gene_sample_counts[i])
