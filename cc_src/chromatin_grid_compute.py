
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

	def compute_bin_counts_sample(self, sample):

		self.plotting_reads = self.gene_reads[self.gene_reads['sample'] == sample]

		x_bin_size = 100
		y_bin_size = 50

		xlims = self.mnase_span
		x_bins = np.arange(xlims[0], xlims[1]+x_bin_size, x_bin_size)
		y_bins = np.arange(0, 200+y_bin_size, y_bin_size)

		self.hist, self.x_edges, self.y_edges = np.histogram2d(self.plotting_reads['mid'], 
			self.plotting_reads['length'], bins=[x_bins, y_bins])

	def plot(self):
		
		xlims = self.mnase_span
		gene = self.gene
		plotting_reads = self.plotting_reads
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 1.5))

		plot_mnase_density(ax1, plotting_reads)
		ax1.set_xticks([])
		ax1.set_xlim(*xlims)

		ax2.imshow(self.hist.T, origin='lower', aspect='auto', cmap='magma_r',
			extent=[self.x_edges[0], self.x_edges[-1], self.y_edges[0], self.y_edges[-1]])
		ax2.set_xlim(*xlims)

		for ax in [ax1, ax2]:
			for x in [gene.TSS-500, gene.TSS, gene.TSS+500]:
				ax.axvline(x, c='green', alpha=0., lw=3)
				
			ax.set_ylim(50, 200)
