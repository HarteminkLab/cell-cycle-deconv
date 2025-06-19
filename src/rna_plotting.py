
# Deprecated, replaced with rna_pileup_plotter.py

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats as stats

# from scipy.stats import norm
# from src.rna_reads import load_rna_reads


# class RNAPlotter:

# 	def __init__(self, gene_window_padding_2):

# 		# TODO: hard-coded...
# 		self.times = [0, 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140]

# 		# Smoothing kernel for RNA curves
# 		self.smooth_kernel = get_smoothing_kernel(100, 5)
# 		self.gene_window_padding_2 = gene_window_padding_2

# 	def set_chrom(self, chr):

# 		# Load all the RNA reads for a chromosome
# 		self.chr = chr
# 		self.chr_rna_reads = load_rna_reads(chr)

# 	def set_gene(self, gene):
# 		"""Set the gene to plot. Will set the plotting window and the RNA-seq reads to plot
# 		in that window"""

# 		chr_rna_reads = self.chr_rna_reads

# 		# Set the gene and gene window
# 		self.gene = gene
# 		self.gene_window = gene['TSS']-self.gene_window_padding_2, \
# 			gene['TSS'] + self.gene_window_padding_2

# 		# Load the rna reads for the gene in the window
# 		chr_rna_reads = self.chr_rna_reads
# 		gene_rna_reads = chr_rna_reads[(chr_rna_reads['start'] > self.gene_window[0]) & 
# 							   (chr_rna_reads['stop'] < self.gene_window[1])]
# 		self.gene_rna_reads = gene_rna_reads
# 		self.strand_pileups = {}
		

# 	def plot_rna_seq_times_strand(self, ax, strand):
# 		"""
# 		For a strand, plot the RNA-seq pileup for all times in the selected window.

# 		Requires computing the pileup for the strand first.
# 		"""

# 		time_pileups = self.strand_pileups[strand]
# 		times = self.times

# 		if strand == '-':
# 			color_map = plt.get_cmap('Blues')
# 		else:
# 			color_map = plt.get_cmap('Reds')

# 		colors = [color_map(i/len(times)) for i in range(len(times))]

# 		for i in range(len(times)):
# 			time = times[i]
# 			sample_pileup = time_pileups[time].copy()

# 			# Compute the log of the pileup
# 			log_sample_pileup = np.log2(sample_pileup+1)

# 			smoothed_log_pileup = np.convolve(
# 				log_sample_pileup.values.reshape(-1), 
# 				self.smooth_kernel, mode='same')

# 			if (strand == '-'): plotting_values = -smoothed_log_pileup
# 			else: plotting_values = smoothed_log_pileup

# 			x_bps = np.arange(self.gene_window[0], self.gene_window[1])
# 			y_pileups = plotting_values

# 			ax.plot(x_bps, y_pileups, c=colors[i])


# 	def compute_pileup_for_strand_all_times(self, strand):
# 		"""
# 		Compute the pileup for all timepoints for a strand in the current window
# 		"""
# 		time_pileups = {}

# 		for time in self.times:
# 			sample_pileup = self.pileup_for_sample(time, strand)
# 			time_pileups[time] = sample_pileup

# 		self.strand_pileups[strand] = time_pileups


# 	def pileup_for_sample(self, time, strand):
# 		"""
# 		Compute the pileup for a selected time and strand
# 		"""
		
# 		gene_window = self.gene_window
# 		gene_rna_reads = self.gene_rna_reads
# 		sample_reads = gene_rna_reads[(gene_rna_reads['strand'] == strand) &
# 									  (gene_rna_reads['sample'] == time)]

# 		gene_window_pileup = pd.DataFrame(index=list(range(*gene_window)))
# 		gene_window_pileup['count'] = 0

# 		for idx, read in sample_reads.iterrows():
# 			gene_window_pileup.loc[read['start']:read['stop']] = \
# 				gene_window_pileup.loc[read['start']:read['stop']]+1

# 		return gene_window_pileup

# 	def plot_rna_seq(self, ax):
# 	    strand = '-'
# 	    self.compute_pileup_for_strand_all_times(strand)

# 	    strand = '+'
# 	    self.compute_pileup_for_strand_all_times(strand)

# 	    self.plot_rna_seq_times_strand(ax, '+')
# 	    self.plot_rna_seq_times_strand(ax, '-')

# 	    ax.set_xlim(self.gene_window[0], self.gene_window[1])
# 	    ax.set_ylim(-10, 10)
# 	    ax.set_xticks([])
# 	    ax.set_yticks([])


# def get_smoothing_kernel(window, std):
#     """Set RNA-seq smoothing kernel"""
#     # smoothing
#     X = np.arange(-1*window/2.0, window/2)
#     return norm.pdf(X, 0, std)
