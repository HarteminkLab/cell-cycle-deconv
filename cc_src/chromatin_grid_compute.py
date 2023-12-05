
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from cc_src.sgd import get_gene_name_orf_name
from cc_src.mnase_plotting import plot_mnase_density


class ChromatinGrid:
	"""
	In this class, we will be taking mnase-seq reads for a gene, and computing a grid of occupancy values for the gene's locus.

	That we can then deconvolve.

	We will be able to plot the gene's locus (raw data) as well as the histogram of the grid for verification that the grid values
	are created properly.

	Notes: We will eventually need to normalize or scale (by copy number and/or by sample depth)
	"""


	def __init__(self):

		self.padding = 1000
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')

	def set_gene(self, gene_name):

		# Get some gene information
		self.orf_name, self.gene_name = get_gene_name_orf_name(gene_name)
		self.gene = self.geneset.loc[self.orf_name]

		self.chr_reads = pd.read_hdf(f'output/mnase/yl_rep2_mnase_reads/yl_rep2_mnase_reads_chr{self.gene.chr}.h5', 
					'mnase_data')

		self.mnase_span = self.gene.TSS-self.padding, self.gene.TSS+self.padding

		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		self.times = self.gene_reads['sample'].unique()

	def compute_bin_counts_sample(self, sample):

		plotting_reads = self.gene_reads[self.gene_reads['sample'] == sample]

		x_bin_size = 100
		y_bin_size = 50

		xlims = self.mnase_span
		x_bins = np.arange(xlims[0], xlims[1]+x_bin_size, x_bin_size)
		y_bins = np.arange(50, 200+y_bin_size, y_bin_size)

		hist, x_edges, y_edges = np.histogram2d(plotting_reads['mid'], 
			plotting_reads['length'], bins=[x_bins, y_bins])

		return plotting_reads, hist, x_edges, y_edges

	def create_bins_per_sample(self):

		samples = self.gene_reads['sample'].unique()

		self.all_plotting_reads = {}
		self.all_hists = {}
		self.all_x_edges = {}
		self.all_y_edges = {}

		for sample in samples:
			plotting_reads, hist, x_edges, y_edges = self.compute_bin_counts_sample(sample)

			self.all_plotting_reads[sample] = plotting_reads
			self.all_hists[sample] = hist.T
			self.all_x_edges[sample] = x_edges
			self.all_y_edges[sample] = y_edges


	def plot(self, ax1, ax2, sample):
		
		xlims = self.mnase_span
		gene = self.gene

		plotting_reads = self.all_plotting_reads[sample]
		hist = self.all_hists[sample]
		x_edges = self.all_x_edges[sample]
		y_edges = self.all_y_edges[sample]

		plot_mnase_density(ax1, plotting_reads)
		ax1.set_xticks([])
		ax1.set_xlim(*xlims)

		ax2.imshow(hist, origin='lower', aspect='auto', cmap='magma_r',
			extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
		ax2.set_xlim(*xlims)

		for ax in [ax1, ax2]:
			for x in [gene.TSS-500, gene.TSS, gene.TSS+500]:
				ax.axvline(x, c='green', alpha=0.75, lw=3)
				
			ax.set_ylim(50, 200)

			xticks = np.arange(gene.TSS-1000, gene.TSS+1500, 500)
			xtick_labels = ['-1000', '-500', 'TSS', '500', '1000']

			ax.set_xticks(xticks)
			ax.set_xticklabels(xtick_labels)
