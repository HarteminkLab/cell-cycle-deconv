
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.utils import print_fl
from src.geneset import get_deconvolved_geneset
from src.config import load_yl_rg1_vst_config
from src.global_config import GlobalConstants


CENTER_BIN = 9 # The +1 lies on the start of the bin index: 9
NUM_BINS_PADDING = 3
PLUS_ONE_BINS = CENTER_BIN-NUM_BINS_PADDING, CENTER_BIN+NUM_BINS_PADDING
PLUS_ONE_RANGE = -NUM_BINS_PADDING*GlobalConstants.BIN_WIDTH, NUM_BINS_PADDING*GlobalConstants.BIN_WIDTH

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

		t_indices = self.config.get_Hpositions_for_branch('t')

		_, _, nuc_bins = len_bins()

		nuc_bins = self.promoter_analysis.strand_corrected_f_images[:, :, \
			nuc_bins[0]:nuc_bins[1]]
		nuc_bins_sum = nuc_bins.sum(axis=2)
		nuc_bins_sum = nuc_bins_sum[:, t_indices]


		# Select the +1 nucleosome bins
		p1_sum = nuc_bins_sum[:, :, PLUS_ONE_BINS[0]:PLUS_ONE_BINS[1]]
		p1_sum = p1_sum.mean(axis=2)

		geneset = self.promoter_analysis.geneset
		gene_plus_one_position = pd.DataFrame(np.zeros((nuc_bins_sum.shape[0], 
			len(t_indices))), index=geneset.index)

		for orf_name, gene in geneset.iterrows():

			gene_nuc_bins = nuc_bins_sum[int(gene.gene_idx)]

			weighted_plus_one_pos, p1_bin_vals = compute_plus_one_movement(gene_nuc_bins, 
				t_indices)

			gene_plus_one_position.loc[orf_name] = weighted_plus_one_pos

		# The +1 nucleosome sum, for filtering out low coverage reads
		# if the +1 cannot be called
		self.nuc_bins_sum = nuc_bins_sum
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

	def show_plus_one_shift_for_gene(self, gene_or_orfname):
		"""Show for a gene, how it's +1 nucleosome is shifting to verify/validate
		the computation of the +1 nucleosome shift calculation"""

		# Get the orf name if the gene name is provided
		from src.sgd import get_gene_name_orf_name, get_gene_title_name
		orf_name, gene_name = get_gene_name_orf_name(gene_or_orfname)
		if gene_name is None: gene_name = orf_name

		gene = self.geneset.loc[orf_name]

		gene_nucs_sum = self.nuc_bins_sum[int(gene.gene_idx)]

		full_extents = GlobalConstants.BIN_EXTENTS

		postG1_indices = self.config.get_Hpositions_for_phase('postG1')
		cg1_indices = self.config.get_Hpositions_for_phase('CG1')
		h_positions = self.config.get_Hpositions_for_branch('t')

		from src.helpers import indices_of_mapping_array
		# The nucleosome array is already in t branch form, so we'll
		# convert the H indices into t indices
		cg1_indices = indices_of_mapping_array(h_positions, cg1_indices)
		postG1_indices = indices_of_mapping_array(h_positions, postG1_indices)

		postG1_ts = self.config.get_phase_timepoints_for_phase('postG1')
		cg1_ts = self.config.get_phase_timepoints_for_phase('CG1')

		# plot cg1 and postg1 separately
		plt.imshow(gene_nucs_sum[cg1_indices], origin='lower', aspect='auto', cmap='magma_r',
		    extent=[full_extents[0], full_extents[1], cg1_ts[0], postG1_ts[0]])
		plt.imshow(gene_nucs_sum[postG1_indices], origin='lower', aspect='auto', cmap='magma_r',
		    extent=[full_extents[0], full_extents[1], postG1_ts[0], postG1_ts[-1]])

		p1_track = self.gene_plus_one_position.loc[orf_name]
		ys = np.concatenate([cg1_ts, postG1_ts])

		plt.scatter(p1_track, ys, c='blue', s=10, marker='D')

		gene_title = get_gene_title_name(orf_name)

		plt.axvline(PLUS_ONE_RANGE[0], c='blue', lw=2, alpha=0.5)
		plt.axvline(PLUS_ONE_RANGE[1], c='blue', lw=2, alpha=0.5)

		from src.helpers import indices_of_mapping_array

		rg1_len, cg1_len, dg1_len = self.config.get_g1_lens()

		h_positions = self.config.get_Hpositions_for_branch('t')
		replication_time = self.repl_w_nuc_max.loc[orf_name].replication_timing

		plt.axhline(replication_time-cg1_len, c='white', lw=1, ls='dotted')

		plt.title(f"+1 nucleosome shift, {gene_title}")
		plt.ylabel("Deconvolved time, mother branch, min")
		plt.xlabel("Genomic position relative to called +1, bp")
		plt.ylim(cg1_ts[0], postG1_ts[-1])


def compute_plus_one_movement(gene_nuc_bins, t_indices):

	from src.global_config import GlobalConstants
	from src.helpers import weighted_mean

	# todo: adjust this logic to handle three bins, currently the +1 lies
	# between two bins, it should lie exactly one a single bin

	bin_width = GlobalConstants.BIN_WIDTH
	bin_width_2 = bin_width//2

	# Padding to see how much the nucleosome shifts
	plus_one_bin_values = gene_nuc_bins[:, PLUS_ONE_BINS[0]:PLUS_ONE_BINS[1]]

	# Define where the bins are in the bp coordinates relative
	# to 0 (+1 location)
	bin_bp_positions = np.arange(-NUM_BINS_PADDING*bin_width, 
		NUM_BINS_PADDING*bin_width, bin_width)+bin_width_2

	weighted_avg_p1_pos = np.apply_along_axis(lambda row: 
		weighted_mean(bin_bp_positions, row), 1, plus_one_bin_values)

	return weighted_avg_p1_pos, plus_one_bin_values

