
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

		# Because, we will be splitting these by chromosome, keep track of the index we should put the
		# gene data into the array
		self.genes['arr_index'] = np.arange(len(self.genes))

		self.window = 10000
		self.n = len(self.genes)
		self.len_span = 0, 250
		self.lengths = np.arange(self.len_span[0], self.len_span[1]+1)
		self.lens = len(self.lengths)-1

	def set_chrom_replicate(self, chromosome, replicate):
		"""Load the MNase-seq data for each chromosome for each gene and bin
		May be a lot of data to hold in memory, so create intermediate files
		for each chromosome"""

		self.chromosome = chromosome
		self.replicate = replicate

		self.chr_genes = self.genes[self.genes.chr == chromosome]
		self.chr_reads = load_mnase_reads(chromosome, replicate)
		self.normalizion_scaling = pd.read_csv(f'datasets/computed_mnase/rep{replicate}_len_counts_fix.csv')
		self.gene_plus_ones = pd.read_csv(f'datasets/computed_mnase/rep{replicate}_plus_ones.csv').set_index('orf_name')
		self.samples = self.chr_reads['sample'].unique()
		self.m = len(self.samples)		

	def create_gene_mnase_histogram(self, timer):

		n = len(self.chr_genes)
		chr_gene_hist_sum = np.zeros((n, self.m, self.lens))

		# Loop through all chromosome genes
		for i in range(n):
			gene = self.chr_genes.iloc[i]
			gene_histogram2d = self.load_gene_read_counts(gene)
			chr_gene_hist_sum[i] = gene_histogram2d.sum(axis=2)

			if (i % 100 == 0) or (i == len(self.chr_genes)-1):
				print(f"{i}/{len(self.chr_genes)} - {timer.get_time()}")

		self.chr_gene_hist_sum = chr_gene_hist_sum

	def normalize(self):

		from src.preprocessing import load_scaling_mat

		scaling_mat = load_scaling_mat(self.replicate).values[:-1, ].T

		# Normalize the histogram
		self.chr_gene_hist_sum_collapsed_x = self.chr_gene_hist_sum
		self.normalized_chr_gene_hist_sum_collapsed_x = self.chr_gene_hist_sum_collapsed_x * scaling_mat[None, :, :]

		# Sum of the normalized histograms
		self.genes_sum_data_normalized = self.normalized_chr_gene_hist_sum_collapsed_x.sum(axis=2)

	def plot_normalized_gene_sums(self):
		plt.figure(figsize=(13, 7))
		plt.imshow(self.genes_sum_data_normalized.T, aspect='auto', cmap='viridis', 
			interpolation=None, origin='lower')
		plt.yticks(np.arange(len(self.samples)), self.samples)
		plt.colorbar()

		plt.title(f"Normalized MNase-seq read counts\n{self.window//1000}kb gene window for chromosome {self.chromosome}"
			+f"\nReplicate {self.replicate}")
		plt.ylabel("Timepoint")
		plt.xlabel(f"Gene in order of position on chromosome {self.chromosome}")


	def load_gene_read_counts(self, gene):
		"""Load the gene reads as a 2D histogram"""

		small_span, mid_span, nuc_span = yl_rep2_len_spans()

		from cc_src.mnase_reads import filter_reads

		chr_reads = self.chr_reads
		plus_one = self.gene_plus_ones.loc[gene.name]['+1']

		window_2 = self.window//2
		gene_span = plus_one-window_2, plus_one+window_2

		# Load the reads for the gene
		gene_sample_counts = np.zeros((self.m, (self.len_span[1]-self.len_span[0]), self.window))

		# For each sample, store into a numpy array the counts at each genomic position
		for i in range(self.m):
			sample = self.samples[i]
			gene_sample_reads = filter_reads(chr_reads, gene_span[0], gene_span[1], sample=sample)
			gene_reads_hist2d = bin_reads(gene_sample_reads, gene_span, self.len_span)
			gene_sample_counts[i] = gene_reads_hist2d

		# Flip if strand is crick
		if gene.strand == '-':
			gene_sample_counts = gene_sample_counts[:, :, ::-1]
		
		return gene_sample_counts


	def plot_genomic_chrom_counts(self):
		"""Plot the summed genomic reads for a chromosome.

		This plot is to debug specifically replicate 1's 60 minute timepoint's oddities.
		"""

		chr_reads = self.chr_reads

		bin_span = 0, chr_reads.mid.max()
		bins = np.arange(0, bin_span[1]+1, 20000)
			
		def get_binned_hist(chr_reads, sample, bins, length_span=None):
			timepoint_reads = chr_reads[chr_reads['sample'] == sample]
			
			if length_span is not None:
				timepoint_reads = timepoint_reads[(timepoint_reads['length'] > length_span[0]) & 
												 (timepoint_reads['length'] < length_span[1])]
				
			counts, _ = np.histogram(timepoint_reads['mid'], bins=bins)
			return counts

		def plot_tps(chr_reads, tps=[50, 60, 70], length_span=(0, 250)):

			for i in range(len(tps)):
				tp = tps[i]
				counts = get_binned_hist(chr_reads, tp, bins)
				x = bins[:-1]
				plt.plot(x, counts, label=f"{tp}")
				plt.xticks(np.arange(0, x[-1], 200000))
				
			plt.title(f"Fragment lengths {length_span}")
			plt.legend()
			plt.xlabel("Genomic position")
			plt.ylabel("Raw counts")
			
		plt.figure(figsize=(16, 3))
		plt.subplots_adjust(top=0.8)

		plt.subplot(1, 3, 1)
		plot_tps(chr_reads)

		plt.subplot(1, 3, 2)
		plot_tps(chr_reads, length_span=(140, 170))

		plt.subplot(1, 3, 3)
		plot_tps(chr_reads, length_span=(0, 100))
		plt.suptitle(f"Replicate {self.replicate}, Chromosome {self.chromosome} reads", fontsize=16)


def bin_reads(reads, span, len_span):
	"""2D binning of MNase-seq reads by length and midpoint"""
	x_bins = np.arange(span[0], span[1]+1)
	y_bins = np.arange(len_span[0], len_span[1]+1)
	counts, _, _ = np.histogram2d(reads['mid'], 
		reads['length'], bins=[x_bins, y_bins])
	return counts.T


def plot_im(ax, gene_reads_hist2d, vmax=0.25):
	"""Plot the histogram for a gene and sample image"""
	ax.imshow(gene_reads_hist2d, origin='lower', cmap='magma_r', vmax=vmax, aspect='auto')
	ax.set_xticks([])
	ax.set_yticks([])

	
def plot_gene_series(analysis, gene_sample_counts):
	""""Plot the entire set of samples of a gene"""

	fig, axs = plt.subplots(analysis.m, 1, figsize=(3, 6))
	plt.subplots_adjust(hspace=0.0)

	for i in range(len(axs)):
		ax = axs[i]
		plot_im(ax, gene_sample_counts[i])
