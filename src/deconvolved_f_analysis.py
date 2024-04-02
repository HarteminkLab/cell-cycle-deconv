
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.sgd import read_sgd_chromosomes
from src.mnase_reads import load_mnase_reads
from src.reference_data import load_plus_ones
from src.reference_data import load_analysis_genes
from src.chromatin_metrics import yl_rep2_len_spans


class DeconvolvedChromatinDataAnalysis:
	"""Class to analyze deconvolved chromatin data, for
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


		self.plus_ones = load_plus_ones()

		# Pre-defined from replicate 1 deconvolution results
		self.chrom_dir = 'output/deconvolve_rep1_g006_2024_02_20/chromatin/'



		# From the chromatin_model.py
		self.prom_len = 288
		self.gb_len = 512

		# For replicate 1 deconvolved run
		self.bin_width = 16


		from src.config import load_yl_rg1_vst_config
		from src.model import Model

		# Get H for the size of F images
		# todo: there's an easier way from the config... requires some refactoring
		config = load_yl_rg1_vst_config(1)
		self.model = Model(config, "CLN1")
		self.H = self.model.H

		# Number of f images
		self.m = self.H.shape[1]


	def set_chrom_replicate(self, chromosome):
		"""Load the MNase-seq data for each chromosome for each gene and bin
		May be a lot of data to hold in memory, so create intermediate files
		for each chromosome"""

		self.chromosome = chromosome

		self.chr_genes = self.genes[self.genes.chr == chromosome]

		self.chrom_length = read_sgd_chromosomes().loc[chromosome]['length']
		# Change chrom length into gene bin width space
		self.chrom_length_in_bins = self.chrom_length // self.bin_width

	def create_gene_sums(self):

		# Create a numpy matrix that whose height is all genes on the chromosome and width is the 
		self.chrom_sum_mat = np.zeros((len(self.chr_genes), self.m, self.chrom_length_in_bins)) + np.nan

		chrom_sum_mat = self.chrom_sum_mat

		for i in range(len(self.chr_genes)):
		    gene = self.chr_genes.iloc[i]
		    gene_f_sums, gene_bin_span = self.get_f_col_sum_for_gene(gene)
		    chrom_sum_mat[i, :, gene_bin_span[0]:gene_bin_span[1]] = gene_f_sums

		gene_f_sums.shape, chrom_sum_mat.shape

		nanmeans = np.nanmean(chrom_sum_mat, axis=0)

		mean_filled_nans = nanmeans.copy()
		mean_filled_nans[np.isnan(mean_filled_nans)] = 0
		self.mean_filled_nans = mean_filled_nans


	def create_binned_sums(self):

		mean_filled_nans = self.mean_filled_nans
		bin_window_size = self.window // self.bin_width
		bin_edges = np.arange(0, self.chrom_length_in_bins, bin_window_size)
		bin_mat = create_binning_matrix(mean_filled_nans.shape[1], bin_edges)
		bin_mat.shape, mean_filled_nans.shape
		binned_counts = mean_filled_nans @ bin_mat.T

		self.binned_counts = binned_counts


	def get_f_col_sum_for_gene(self, gene):
		"""Get the F image for the gene
		
		Return the sum its columns and its position on the chromosome (in binned-space)
		"""
		
		orf_name = gene.name

		f_filepath = find_f_filepath(self.chrom_dir, orf_name)
		f = np.load(f_filepath)

		plus_one_loc = self.plus_ones.loc[orf_name]['+1']

		if gene.strand == "+":
			gene_span = plus_one_loc-self.prom_len, plus_one_loc+self.gb_len
		else:
			gene_span = plus_one_loc-self.gb_len, plus_one_loc+self.prom_len

		gene_span_in_bins = gene_span[0] // self.bin_width, gene_span[1] // self.bin_width

		current_f = f
		current_f_summed_columns = current_f.sum(axis=1)
		gene_span_in_bins = int(gene_span_in_bins[0]), int(gene_span_in_bins[1])
		
		return current_f_summed_columns, gene_span_in_bins


	def get_chrom_counts(self):

		for i in range(len(chr_genes)):
			gene = chr_genes.iloc[i]
			
			gene_f_sums, gene_bin_span = get_f_col_sum_for_gene(gene, chrom_dir, f_index)
			chrom_sum_mat[i, gene_bin_span[0]:gene_bin_span[1]] = gene_f_sums


	def plot_sums(self):
		fig = plt.figure(figsize=(18, 9))
		plt.imshow(self.binned_counts, vmax=10000,
    		extent=[0, self.chrom_length, 0, self.m], origin='lower', aspect='auto')
		plt.title(f"Chr {self.chromosome}")
		return fig

def bin_counts(counts, window_size):
	"""
	Bins an array of counts into non-overlapping windows of a specified size and sums the counts within each window,
	including a final window that may be smaller if the total count is not a multiple of the window size.

	Parameters:
	- counts: numpy array of counts.
	- window_size: int, the size of the window to bin the counts into.

	Returns:
	- numpy array of summed counts for each window.
	"""
	
	# Calculate the number of full windows
	full_windows = len(counts) // window_size
	
	# Initialize an array to hold the sum for each window
	window_sums = np.zeros(full_windows + (len(counts) % window_size > 0))
	
	# Sum each full window
	for i in range(full_windows):
		start_index = i * window_size
		window_sums[i] = np.sum(counts[start_index:start_index + window_size])
	
	# Handle the partial window if exists
	if len(counts) % window_size > 0:
		start_index = full_windows * window_size
		window_sums[-1] = np.sum(counts[start_index:])
	
	return window_sums


def create_binning_matrix(counts_length, bin_edges):
    """
    Create a matrix that will allow us to bin vectors of shape (counts_length) into
    a matrix with defined bin locations (edges).

    The use of this will be that the counts can be a matrix and we will be able to 
    bin multiple timepoints at once with the resulting matrix
    """
    num_bins = len(bin_edges) - 1
    binning_matrix = np.zeros((num_bins, counts_length))
    
    for i in range(num_bins):
        start_edge = bin_edges[i]
        end_edge = bin_edges[i + 1]
        
        # Assuming counts indices map directly to the bin ranges specified by start_edge and end_edge
        binning_matrix[i, start_edge:end_edge] = 1
    
    return binning_matrix


def find_f_filepath(chrom_dir, orf_name):
	pattern = os.path.join(chrom_dir, f'*_f_{orf_name}*')
	found_files = glob.glob(pattern)
	if len(found_files) == 0: 
		print(f"Could not find f file for {orf_name}")
		return None
	return found_files[0]
