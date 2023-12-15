
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cc_src.orf_plotter import ORFAnnotationPlotter
from cc_src.rna_plotting import RNAPlotter
from cc_src.mnase_plotting import MNasePlotter


class GeneLocusPlotter:
	"""
	Class to plot the raw data and the deconvolution for each gene
	we will use this class to plot all genes on a chromosome. We will also 
	define the loop that plots these fits across all chromosomes for all genes.
	"""

	def __init__(self):

		self.gene_window_padding_2 = 1000
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')
		self.chr = None

		# Individual plotters
		self.orf_plotter = ORFAnnotationPlotter(self.geneset)
		self.chr_rna_plotter = RNAPlotter(self.gene_window_padding_2)
		self.chr_mnase_plotter = MNasePlotter(self.gene_window_padding_2)
		self.title = None

	def set_chrom(self, chrom):

		# Per chromosome config
		if self.chr != chrom:
			self.chr = chrom
			self.chr_mnase_plotter.set_chrom(chrom)
			self.chr_rna_plotter.set_chrom(chrom)

	def set_gene(self, gene_name_or_orf_name, normalize_mnase=True):

		from cc_src.sgd import get_orfname

		if gene_name_or_orf_name in self.geneset.index.values:
			orfname = gene_name_or_orf_name
		else:
			orfname = get_orfname(gene_name_or_orf_name)

		# Load a sample gene
		gene = self.geneset.loc[orfname]
		self.gene = gene
		self.gene_window = self.gene.TSS-self.gene_window_padding_2, self.gene.TSS+self.gene_window_padding_2

		# Set the chromosome to load the correct reads
		if self.chr != gene.chr:
			self.set_chrom(gene.chr)

		# Configure the ORF plotter
		self.orf_plotter.set_span_chrom(self.gene_window, gene.chr)
		self.chr_mnase_plotter.set_gene(gene, normalize_mnase)
		# self.chr_rna_plotter.set_gene(gene)


	def plot(self):
		# number of rows: orfs, rna-seq, mnase_reads (num of timepoints)
		rows = 2 + len(self.chr_rna_plotter.times)

		fig, axs = plt.subplots(rows, 1, figsize=(9, 16))
		axs = np.array(axs).flatten()

		orfs_ax = axs[0]
		rna_ax = axs[1]

		self.orf_plotter.plot_orf_annotations(orfs_ax)
		# self.chr_rna_plotter.plot_rna_seq(rna_ax)
		self.chr_mnase_plotter.plot(axs[2:])

		# plot TSS
		for ax in axs:
			ax.axvline(self.gene.TSS, c='black', lw=1, linestyle='solid', alpha=0.5)


		if self.title is None:
			title = f"{self.gene.gene} / {self.gene.name} - "\
				f"{self.gene.chr}: {self.gene_window[0]}-{self.gene_window[1]}"
		else:
			title = self.title

		plt.suptitle(title)

		return fig


class DeconvolutionPlotter:

	def __init__(self):
		from src.load_deconvolutions import load_yl_rep2_gene_body_nucleosome_entropy_deconvolution

		self.model, self.ptrs = load_yl_rep2_gene_body_nucleosome_entropy_deconvolution()

	def plot_gene(self, orfname_or_genename, title):
		fig = self.model.plot_deconvolved_gene(orfname_or_genename, 
			self.model.deconv_f, self.model.deconv_g, title=title)
		return fig

