
import cvxpy
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.sgd import get_gene_name_orf_name
from src.mnase_plotting import plot_mnase_density

from src.deconvolve_chromatin import deconvolve_chromatin
from src.model import Model
from src.timer import Timer


class ComputePlusOne:
	"""
	Find the +1 nucleosome location for a gene
	"""


	def __init__(self):

		# Padding defines the window around the TSS to retrieve MNase data
		self.padding = 1000
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')


	def load_mnase_gene(self, gene_name, replicate, log=True):

		# Get some gene information
		self.computed_plus_one = None
		self.orf_name, self.gene_name = get_gene_name_orf_name(gene_name)
		self.gene = self.geneset.loc[self.orf_name]
		self.mnase_span = self.gene.TSS-self.padding, self.gene.TSS+self.padding

		if log:
			print(f"Loading MNase reads for {gene_name}...", end='')

		# TODO: This may take a little while, when we've deconvolved already we may want to skip this step,
		# But that will mean needing to save the +1 location to disk.
		self.chr_reads = pd.read_hdf(f'output/mnase/yl_rep{replicate}_mnase_reads/yl_rep{replicate}_mnase_reads_chr{self.gene.chr}.h5', 
					'mnase_data')
		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]
		self.find_max_plusOne_pos()


	def find_max_plusOne_pos(self):
		"""
		Find the position of the +1 by finding the max number of nucleosome reads in
		a 200bp window around the TSS.

		For the currently selected gene
		"""
		from src.global_config import fragment_lengths_definitions
		small_lens, med_lens, nuc_lens = fragment_lengths_definitions()

		# Next, we will align at the +1
		# from the TSS, stack up all timepoints, then look up and dowstream (200 bp window) for the
		# position with the highest number of reads, and we will use that position as our +1 position
		# We will put that position into our gene data set and use that as our reference data set

		# We can get all of the  nucleosome length fragments for the gene, and stack them up by time

		cur_reads = self.gene_reads.copy()

		# Search around the TSS with a 200bp window
		window = 200
		search_peak_span = self.gene.TSS-window//2, \
			self.gene.TSS+window//2 

		cur_nuc_reads = cur_reads[(cur_reads['length'] >= nuc_lens[0]) & 
							  (cur_reads['length'] < nuc_lens[1]) & 
								 (cur_reads['mid'] >= search_peak_span[0]) &
								 (cur_reads['mid'] < search_peak_span[1])]

		if len(cur_nuc_reads) == 0:
			self.computed_plus_one = np.nan
			return np.nan

		counts_per_pos_search = cur_nuc_reads.groupby('mid').count()
		counts_per_pos_search = counts_per_pos_search[['start']].rename({'start': 'count'})
		pos_max = counts_per_pos_search.idxmax().start

		self.computed_plus_one = pos_max

		return pos_max

