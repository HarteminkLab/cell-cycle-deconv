
import cvxpy as cp

import pandas as pd
import numpy as np

from src.helpers import calcH
from matplotlib import pyplot as plt
from src.utils import print_fl
from src.global_config import GlobalConstants
from src.config import load_yl_rg1_vst_config
from src.delta_config import load_yl_delta_config
from src.geneset import get_deconvolved_geneset


class StepReplicationChromatinDeconvolveSolver:
	"""
	Compute an estimate for a single point in the deconvolution timing profiles in which
	replication occurs.
	"""

	def __init__(self, mnase_analysis_rep1, mnase_analysis_rep2):

		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.mnase_analysis_rep2 = mnase_analysis_rep2

		self.config1 = load_yl_delta_config(1)
		self.config2 = load_yl_delta_config(2)

		self.H1, _ = calcH(self.config1.intervals_wt1, 
			GlobalConstants.CHROM_WT1_TIMEPOINTS)

		self.H2, _ = calcH(self.config2.intervals_wt1, 
			GlobalConstants.CHROM_WT2_TIMEPOINTS)

		self.H = np.concatenate([self.H1, self.H2])

		self.geneset = get_deconvolved_geneset()


	def set_chrom(self, chrom):
		"""Select the bins for the given chromosome"""

		self.chrom = chrom

		bin_curves1 = self.mnase_analysis_rep1.normalized_bin_curves
		bin_curves2 = self.mnase_analysis_rep2.normalized_bin_curves

		self.chr_bin_curves1 = bin_curves1.loc[chrom]
		self.chr_bin_curves2 = bin_curves2.loc[chrom]

		self.adjust_bins_for_anomalous_min_maxes()

	# use copy number dataset to determine range of values
	def scale_bins(self, bins, replicate):

		copy_num_rep = load_copy_num(replicate)
		# the values of g are normalized to be between 0-1.
		# So normalize them to be within the range of the copy number
		# values. 
		# todo: Should rethink how the bins should be normalized.
		#       As they should reflect the actual copy number of the sample
		#       including the halted cells proportion, meaning
		#       the max will never actually get to 2.0 in the experiment.
		copy_min, copy_max = copy_num_rep.min().scale, copy_num_rep.max().scale
		scale_g = (copy_max-copy_min)
		bins_normalized = bins * scale_g + copy_min
		return bins_normalized

	def select_bins(self, bins):

		from src.sgd import get_orfname

		bins1 = self.adjusted_curves_1.iloc[bins].values.T
		bins2 = self.adjusted_curves_2.iloc[bins].values.T

		# Normalize by the replicate's copy number
		# rather than 0-1, ~1-1.5, (handles number of halted cells)
		self.g1 = self.scale_bins(bins1, 1)
		self.g2 = self.scale_bins(bins2, 2)

		# Concatenate for the combined model
		self.g = np.concatenate([self.g1, self.g2])


	def solve(self, verbose=False):

		g = self.g
		H = self.H

		n, m = H.shape
		n, u = self.g.shape

		config = self.config1
		transition_point = cp.Variable(integer=True)

		f_delta_i = config.get_Hpositions_for_phase('Delta')
		f_rg1_i = config.get_Hpositions_for_phase('RG1')
		f_cg1_i = config.get_Hpositions_for_phase('CG1')
		f_pg1_i = config.get_Hpositions_for_phase('postG1')

		f_rg1 = np.zeros((len(f_rg1_i), u)).astype(bool)
		f_cg1 = np.zeros((len(f_cg1_i), u)).astype(bool)
		f_delta = np.zeros((len(f_delta_i), u)).astype(bool)
		f_pg1 = cp.Variable((len(f_pg1_i), u), boolean=True)
		f_halted = np.zeros((1, u)).astype(bool)

		# F is vertical stack of 0s for all of the G1s, the
		# Post G1 boolean vector we are searching for, and a 0 for halted
		# F will be converted to 1+ values in the objective
		# and the final solution.
		f = cp.vstack([f_rg1, f_cg1, f_delta, f_pg1, f_halted])+1

		elementwise_result = H@f - g

		objective = cp.Minimize(
			cp.sum(cp.norm(elementwise_result, 'fro')**2)
		)

		constraints = []
								
		# Post G1
		for i in range(1, len(f_pg1_i)):
			index = f_pg1_i[i]
			prev_index = f_pg1_i[i-1]
			constraints.append(f[index] >= f[prev_index])

		problem = cp.Problem(objective, constraints)
		problem.solve(verbose=verbose, solver=cp.MOSEK)

		if verbose:
			print("CVXPY problem finished with status: ", problem.status)

		self.f = f.value

	def compute_replication_timing(self, f):
		replication_timing_indices = (f > 1.5).argmax(axis=0)
		replication_timing = [self.config1.get_timepoint_for_index(i) 
			for i in replication_timing_indices]
		self.replication_timing_df = pd.DataFrame(data={
			'H_index': replication_timing_indices,
			'replication_time': replication_timing
			})

	def plot_result(self):
		plt.figure(figsize=(13, 2))

		plt.subplot(1, 3, 1)
		plt.plot(self.f)

		plt.subplot(1, 3, 2)
		plt.plot(self.g, c='black')
		plt.plot(self.H@self.f, c='red')


	def adjust_bins_for_anomalous_min_maxes(self):
		"""Adjust bins for situations in which the min in the second half
		of the raw data is less than the first half. Possibly due to alpha
		factor changes.

		Record the bins/genes in which this occurs.
		"""
		adjusted_curves_1 = self.chr_bin_curves1.copy()
		adjusted_curves_2 = self.chr_bin_curves2.copy()

		fixed_genes = []

		for bin_id in np.arange(len(self.chr_bin_curves1)):

			bin_dat, normalized_bin_dat, copy_scaling, (first_half, second_half),\
				(normalized_first, normalized_second) = \
				self.normalize_raw_bin_by_half_copy_scaling(bin_id, replicate=1)\

			adjusted_curves_1.iloc[bin_id] = normalized_bin_dat

			bin_dat, normalized_bin_dat, copy_scaling, (first_half, second_half),\
				(normalized_first, normalized_second) = \
				self.normalize_raw_bin_by_half_copy_scaling(bin_id, replicate=2)\

			adjusted_curves_2.iloc[bin_id] = normalized_bin_dat

		self.adjusted_curves_1 = adjusted_curves_1
		self.adjusted_curves_2 = adjusted_curves_2


	def explore_adjustment_and_normalization(self, bin_idx, rep):
		"""
		Plot to check how normalization and any adjustments need to be made
		to a raw bin.

		This procedure is helpful in identifying bins whose occupancy values
		are affected by alpha factor occupancy values. And for thinking and
		executing ideas around adjustments needed for these bins to compute
		an appropriate replicating timing.
		"""

		# Get the first cell cycle end, to partition the raw data
		# into two parts
		recovery_len, first_s_start, first_s_end, first_cc_end = \
			self.config1.get_key_timepoints_in_raw()

		bin_dat, normalized_bin_dat, copy_scaling, (first_half, second_half),\
			(normalized_first, normalized_second) = \
			self.normalize_raw_bin_by_half_copy_scaling(bin_idx, rep)
		plt.figure(figsize=(11, 2))

		plt.subplot(1, 3, 1)
		plt.plot(copy_scaling)
		plt.axvline(first_cc_end, c='red')
		plt.title("Copy number scaling")

		plt.subplot(1, 3, 2)
		plt.plot(bin_dat, lw=1, ls='dotted', c='black')
		plt.axvline(first_cc_end, c='red')

		plt.plot(first_half, c='blue')
		plt.plot(second_half, c='red')
		ax = plt.subplot(1, 3, 3)
		plt.title("First-second half raw")

		plt.plot(normalized_bin_dat, c='black', lw=1, ls='dotted')
		plt.plot(first_half.index, normalized_first)
		plt.plot(second_half.index, normalized_second)
		plt.title("Normalized")

		plt.subplots_adjust(top=0.8)
		plt.suptitle(f"Bin {bin_idx}, Rep{rep}")
		
	def normalize_raw_bin_by_half_copy_scaling(self, bin_id, replicate):
		
		if replicate == 1:
			bin_dat = self.chr_bin_curves1.iloc[bin_id]
			copy_scaling = load_copy_num(1)
			recovery_len, first_s_start, first_s_end, first_cc_end = \
				self.config1.get_key_timepoints_in_raw()
		else:
			bin_dat = self.chr_bin_curves2.iloc[bin_id]
			copy_scaling = load_copy_num(2)
			recovery_len, first_s_start, first_s_end, first_cc_end = \
				self.config2.get_key_timepoints_in_raw()

		# Normalize copy scaling vector to the same as how the 
		# bins were normalized. 
		# todo: we may want to refactor/rethink this normalization process
		# for all bins..
		from src.TracerPlotter import normalize_max_min
		copy_scaling_normalized = normalize_max_min(copy_scaling.values)
		copy_scaling.loc[:] = copy_scaling_normalized

		# Example bin: 50 in chr10
		# It also appears that there is a signal of recovery enrichment
		# from 0-30 minutes in the raw data.

		first_half_indices = bin_dat.index <= first_cc_end
		second_half_indices = bin_dat.index > first_cc_end+10

		first_half = bin_dat.loc[first_half_indices]
		second_half = bin_dat.loc[second_half_indices]

		first_half_copy_num = copy_scaling[first_half_indices]
		second_half_copy_num = copy_scaling[second_half_indices]
		cp_first_half_min, cp_first_half_max = first_half_copy_num.min().scale, \
			first_half_copy_num.max().scale

		cp_second_half_min, cp_second_half_max = second_half_copy_num.min().scale, \
			second_half_copy_num.max().scale

		def normalize_to_a_b(dat, min_a, max_b):
			"""Normalize a vector to match a given min and max values.
			"""
			dat = dat.copy()
			dat = (dat - dat.min()) / (dat.max() - dat.min())
			dat = dat*(max_b-min_a)+min_a
			return dat

		normalized_first = normalize_to_a_b(first_half, 
			cp_first_half_min, cp_first_half_max)
		normalized_second = normalize_to_a_b(second_half, 
			cp_second_half_min, cp_second_half_max)

		# Because we are normalizing by two disjointed curves
		# Interpolate the middle point as the middle point between the two
		normalized_bin_dat = bin_dat.copy()
		normalized_bin_dat.loc[first_half_indices] = normalized_first
		normalized_bin_dat.loc[second_half_indices] = normalized_second

		middle_pt = (normalized_first.iloc[-1]+normalized_second.iloc[0])/2.
		normalized_bin_dat.iloc[len(normalized_first)] = middle_pt
		
		return bin_dat, normalized_bin_dat, copy_scaling, \
			(first_half, second_half), (normalized_first, normalized_second)


def load_copy_num(replicate):
	copy_num_file = f'datasets/computed_mnase/dna_copy_scaling_rep{replicate}.csv'
	copy_num_rep = pd.read_csv(copy_num_file).set_index("Unnamed: 0")
	return copy_num_rep
