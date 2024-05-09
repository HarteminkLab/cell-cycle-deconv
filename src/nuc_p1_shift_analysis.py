
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.utils import print_fl
from src.geneset import get_deconvolved_geneset
from src.config import load_yl_rg1_vst_config


CENTER_BIN = 9 # The +1 lies on the start of the bin index: 9
NUM_BINS_PADDING = 2


class ComputePlusOneShiftAnalysis:
	"""
	Compute the shift of the plus one nucleosome in the chromatin.
	"""

	def __init__(self, promoter_analysis):

		self.geneset = promoter_analysis.geneset
		self.promoter_analysis = promoter_analysis

		# Load the replicate 1 config for phase/branch indices
		from src.config import load_yl_rg1_vst_config
		config = load_yl_rg1_vst_config(1)
		self.config = config

	def compute_p1_shift(self):
		from src.chromatin_metrics import len_bins

		_, _, nuc_bins = len_bins()

		nuc_bins = self.promoter_analysis.strand_corrected_f_images[:, :, \
			nuc_bins[0]:nuc_bins[1]]
		nuc_bins_sum = nuc_bins.sum(axis=2)

		t_indices = self.config.get_Hpositions_for_branch('t')

		# Select the +1 nucleosome bins
		plus_one_bins = CENTER_BIN-NUM_BINS_PADDING, CENTER_BIN+NUM_BINS_PADDING
		p1_sum = nuc_bins_sum[:, t_indices, plus_one_bins[0]:plus_one_bins[1]]
		p1_sum = p1_sum.mean(axis=2)

		geneset = self.promoter_analysis.geneset
		gene_plus_one_position = pd.DataFrame(np.zeros((nuc_bins_sum.shape[0], 
			len(t_indices))), index=geneset.index)

		for orf_name, gene in geneset.iterrows():
			weighted_plus_one_pos, p1_bin_vals = compute_plus_one_movement(gene, 
				nuc_bins_sum, t_indices)
			gene_plus_one_position.loc[orf_name] = weighted_plus_one_pos

		# The +1 nucleosome sum, for filtering out low coverage reads
		# if the +1 cannot be called
		self.p1_sum = p1_sum
		self.gene_plus_one_position = gene_plus_one_position

	def normalize_plus_ones(self):
		self.p1_mean_only_norm = self.gene_plus_one_position.copy()
		p1_values = self.gene_plus_one_position.values
		p1_values = p1_values - p1_values.mean(axis=1).reshape((-1, 1))
		self.p1_mean_only_norm[:] = p1_values

	def join_with_replication_timing(self):
		"""Join metadata with replication timing to create a dataset that allows us
		to filter out low nuc coverage genes when comparing replication timing genes"""

		# Create metadata of the maximum nucleosome bins occupancy
		# and the max change in shift (absolute)
		nuc_bins_max_df = self.geneset[[]].copy()
		nuc_bins_max_df['bin_max'] = self.p1_sum.max(axis=1)
		nuc_bins_max_df['max_difference_shift'] = np.abs(self.p1_mean_only_norm.max(axis=1))
		nuc_bins_max_df['gene_idx'] = np.arange(len(nuc_bins_max_df))

		# Join with the replication data for concordant analysis
		geneset_repl = pd.read_csv('datasets/computed_mnase/all_repl_timing_deconvolved_w_metadata_2024_05_09.csv')
		geneset_repl = geneset_repl.set_index('orf_name')
		repl_w_nuc_max = geneset_repl.join(nuc_bins_max_df)

		self.repl_w_nuc_max = repl_w_nuc_max


	def save_plus_ones(self, save_dir):
		self.gene_plus_one_position.to_csv(f'{save_dir}/computed_plus_one_movement.csv')
		self.p1_mean_only_norm.to_csv(f'{save_dir}/computed_plus_one_movement_meannorm.csv')
		self.repl_w_nuc_max.to_csv(f'{save_dir}/p1_meta_data.csv')


def compute_plus_one_movement(gene, nuc_bins_sum, t_indices):

	from src.global_config import GlobalConstants
	from src.helpers import weighted_mean

	# todo: adjust this logic to handle three bins, currently the +1 lies
	# between two bins, it should lie exactly one a single bin

	bin_width = GlobalConstants.BIN_WIDTH
	bin_width_2 = bin_width//2

	gene_nuc_bins = nuc_bins_sum[int(gene.gene_idx)]

	# Padding to see how much the nucleosome shifts
	plus_one_bins = CENTER_BIN-NUM_BINS_PADDING, CENTER_BIN+NUM_BINS_PADDING
	plus_one_bin_values = gene_nuc_bins[t_indices, plus_one_bins[0]:plus_one_bins[1]]

	# Define where the bins are in the bp coordinates relative
	# to 0 (+1 location)
	bin_bp_positions = np.arange(-NUM_BINS_PADDING*bin_width, 
		NUM_BINS_PADDING*bin_width, bin_width)+bin_width_2

	weighted_avg_p1_pos = np.apply_along_axis(lambda row: 
		weighted_mean(bin_bp_positions, row), 1, plus_one_bin_values)

	return weighted_avg_p1_pos, plus_one_bin_values

