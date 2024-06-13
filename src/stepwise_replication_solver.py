
import cvxpy as cp

import pandas as pd
import numpy as np

from src.helpers import calcH
from src.calcH_single_g1 import calcH as calcH_single_g1
from matplotlib import pyplot as plt
from src.utils import print_fl
from src.global_config import GlobalConstants
from src.geneset import get_deconvolved_geneset


from src.delta_config import Config as DeltaConfig
from src.single_G1_config import Config as SharedConfig
from src.config import Config as DistinctConfig

class StepReplicationChromatinDeconvolveSolver:
	"""
	Compute an estimate for a single point in the deconvolution timing profiles in which
	replication occurs.
	"""

	def __init__(self, mnase_analysis_rep1, mnase_analysis_rep2, config_type="delta"):

		self.config_type = config_type

		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.mnase_analysis_rep2 = mnase_analysis_rep2

		self.config1, self.config2 = load_configs_by_config_type(config_type)
		calcH_func = self.config1.calcH_function

		print(f"Deconvolving with config: {config_type}, {type(self.config1)}, {calcH_func}")

		self.H1, _ = calcH_func(self.config1.intervals_wt1, 
			GlobalConstants.CHROM_WT1_TIMEPOINTS)

		self.H2, _ = calcH_func(self.config2.intervals_wt1, 
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

		# F is vertical stack of 0s for all of the G1s, the
		# Post G1 boolean vector we are searching for, and a 0 for halted
		# F will be converted to 1+ values in the objective
		# and the final solution.
		if isinstance(config, DeltaConfig):
			f_delta_i = config.get_Hpositions_for_phase('Delta')
			f_rg1_i = config.get_Hpositions_for_phase('RG1')
			f_cg1_i = config.get_Hpositions_for_phase('CG1')
			f_pg1_i = config.get_Hpositions_for_phase('postG1')

			f_rg1 = np.zeros((len(f_rg1_i), u)).astype(bool)
			f_cg1 = np.zeros((len(f_cg1_i), u)).astype(bool)
			f_delta = np.zeros((len(f_delta_i), u)).astype(bool)
			f_pg1 = cp.Variable((len(f_pg1_i), u), boolean=True)
			f_halted = np.zeros((1, u)).astype(bool)

			f = cp.vstack([f_rg1, f_cg1, f_delta, f_pg1, f_halted])+1

		elif isinstance(config, SharedConfig):
			f_rg1_i = config.get_Hpositions_for_phase('RG1')
			f_cg1_i = config.get_Hpositions_for_phase('CG1')
			f_pg1_i = config.get_Hpositions_for_phase('postG1')

			f_rg1 = np.zeros((len(f_rg1_i), u)).astype(bool)
			f_cg1 = np.zeros((len(f_cg1_i), u)).astype(bool)
			f_pg1 = cp.Variable((len(f_pg1_i), u), boolean=True)
			f_halted = np.zeros((1, u)).astype(bool)

			f = cp.vstack([f_rg1, f_cg1, f_pg1, f_halted])+1

		elif isinstance(config, DistinctConfig):
			f_dg1_i = config.get_Hpositions_for_phase('DG1')
			f_rg1_i = config.get_Hpositions_for_phase('RG1')
			f_cg1_i = config.get_Hpositions_for_phase('CG1')
			f_pg1_i = config.get_Hpositions_for_phase('postG1')

			f_rg1 = np.zeros((len(f_rg1_i), u)).astype(bool)
			f_cg1 = np.zeros((len(f_cg1_i), u)).astype(bool)
			f_dg1 = np.zeros((len(f_dg1_i), u)).astype(bool)
			f_pg1 = cp.Variable((len(f_pg1_i), u), boolean=True)
			f_halted = np.zeros((1, u)).astype(bool)

			f = cp.vstack([f_rg1, f_cg1, f_dg1, f_pg1, f_halted])+1

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

	from src.timer import Timer

	def deconvolve_all_chromosomes(self, chroms=range(1, 17)):

		from src.timer import Timer

		timer = Timer()
		all_chrom_fs = pd.DataFrame()
		self.all_chrom_fs_df = all_chrom_fs

		for chrom in chroms:
			print(f"Chromosome {chrom}")
			self.deconvolve_chr_all_bins(chrom)
			current_chr_fs_df = self.chr_fs_df.copy()
			current_chr_fs_df['chr'] = chrom
			current_chr_fs_df = current_chr_fs_df.reset_index().rename(
				columns={'index': 'start'}).set_index(['chr', 'start'])
			all_chrom_fs = pd.concat([all_chrom_fs, current_chr_fs_df])
			timer.print_time(f"done.")

		self.all_chrom_fs_df = all_chrom_fs

	def deconvolve_chr_all_bins(self, chrom):

		self.set_chrom(chrom)
		from src.timer import Timer

		timer = Timer()
		n = len(self.chr_bin_curves1)
		all_fs = None

		for bin_idx in np.arange(n):
			
			# Total counts in this bin is 0, skip
			if (not self.chr_bin_curves1.iloc[bin_idx].sum() > 0):
				continue

			self.select_bins([bin_idx])

			try:
				self.solve(verbose=False)
			except:
				print(f"Error with bin: {bin_idx}, skipping.")
				continue

			if bin_idx % 40 == 0:
				timer.print_time(f"{bin_idx}/{n}")
			current_f = self.f
			
			if all_fs is None:
				all_fs = np.zeros((n, current_f.shape[0]))

			all_fs[bin_idx] = current_f.flatten()

		all_fs[all_fs == 0] = np.nan
		all_fs_df = pd.DataFrame(all_fs, index=self.chr_bin_curves1.index)
		self.chr_fs_df = all_fs_df

	def plot_deconvolved_chrom_f(self):

		from src.sgd import get_chromosome_length

		postg1_indices = self.config1.get_Hpositions_for_phase('postG1')
		postg1_tps = self.config1.get_phase_timepoints_for_phase('postG1')

		g1lens = self.config1.get_g1_lens()
		cg1_len = g1lens[1]

		chrom_len = get_chromosome_length(self.chrom)
		plt.figure(figsize=(9, 2))

		postg1_f = self.chr_fs_df[postg1_indices]

		plt.imshow(postg1_f.T, origin='lower', aspect='auto', 
				   extent=[0, postg1_f.index[-1], postg1_tps[0]+cg1_len, postg1_tps[-1]+cg1_len],
				  vmin=1, vmax=2, cmap='inferno')
		plt.title(f"Chromosome {self.chrom}, replicating timing profile", 
			fontsize=13, pad=10)
		plt.yticks(np.arange(20, 40, 5))

		plt.ylim(40, 23)
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Replicating timing, min")


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

	def create_replication_timing_indices(self):
		chroms = self.all_chrom_fs_df.index.get_level_values(0).unique()
		self.all_chrs_replication_profile = convert_to_replication_timing(self.all_chrom_fs_df, 
														 chroms=chroms)

	def compute_gene_replication_timing(self):
		from src.mnase_replication_timing_analysis import get_bin_for_position
		from src.timer import Timer
		from src.geneset import get_deconvolved_geneset

		genes = get_deconvolved_geneset()
		repl_profile = self.all_chrs_replication_profile
		H = self.H
		config = self.config1
			
		repl_timing = genes[[]].copy()
		repl_timing['replication_time'] = np.nan
		repl_timing['replication_H_index'] = np.nan

		chroms = self.all_chrs_replication_profile.index.get_level_values(0).unique()

		for orf_name, gene in genes.iterrows():

			if gene.chr not in chroms: continue

			start_indices = repl_profile.loc[gene.chr].index.values
			chrom_repl_profile = repl_profile.loc[gene.chr]
			bin_idx, bin_start_bp = get_bin_for_position(gene.TSS, start_indices)

			replication_index = chrom_repl_profile.loc[bin_start_bp].values[0]

			repl_timing.loc[orf_name, 'replication_H_index'] = replication_index
			repl_timing.loc[orf_name, 'replication_time'] = config.get_timepoint_for_index(replication_index)

		self.gene_replication_timing = repl_timing

	def save_replication_timings(self, directory):

			save_path = f"{directory}/chrom_replication_timing_{self.config_type}.csv"
			self.all_chrs_replication_profile.to_csv(save_path)
			print(f"Saved chromosome replication timingn to {save_path}")

			save_path = f"{directory}/genes_replication_timing_{self.config_type}.csv"
			self.gene_replication_timing.to_csv(save_path)
			print(f"Saved gene replication timingn to {save_path}")


def load_copy_num(replicate):
	copy_num_file = f'datasets/computed_mnase/dna_copy_scaling_rep{replicate}.csv'
	copy_num_rep = pd.read_csv(copy_num_file).set_index("Unnamed: 0")
	return copy_num_rep


def convert_to_replication_timing(repl_profile, chroms=range(1, 17), interpolate=True):

	replication_profile = pd.DataFrame()

	for chrom in chroms:

		repl_prof_values = repl_profile.loc[chrom].idxmax(axis=1).astype(float)

		if interpolate:
			interpolated_values = interpolate_values(repl_prof_values)
			repl_prof_w_interpolation = repl_prof_values.copy()
			repl_prof_w_interpolation.loc[np.isnan(repl_prof_values)] = interpolated_values.values.flatten()
			repl_prof_values = repl_prof_w_interpolation

		chrom_repl_prof = pd.DataFrame({'start': repl_prof_values.index,
			'replication_index': repl_prof_values.values})
		chrom_repl_prof['chr'] = chrom

		replication_profile = pd.concat([replication_profile, chrom_repl_prof])

	replication_profile = replication_profile.set_index(['chr', 'start'])

	if interpolate:
		replication_profile = replication_profile.astype(int)

	return replication_profile


def load_gene_replication_profile(config_type):
	path = f'data/replication_timing/yl_2019/genes_replication_timing_{config_type}.csv'
	replication_profile = pd.read_csv(path).set_index('orf_name')
	return replication_profile


def interpolate_values(repl_prof_values, step=2000):

	# Loop through the replication profile values, when a string of nans are reached
	# record the start and end, then interpolate between before the start and after the end
	k = len(repl_prof_values.values)

	startna = None
	lastna = None
	interpolate_start = np.nan
	interpolate_end = np.nan

	interpolate_tuples = []

	for i in range(0, k):
		prior_index = repl_prof_values.index[i-1]
		prior_value = repl_prof_values.values[i-1]
		index = repl_prof_values.index[i]
		value = repl_prof_values.values[i]

		if np.isnan(value):
			if startna is None:
				startna = index
				interpolate_start = prior_value
		elif not startna is None:
			lastna = prior_index
			interpolate_end = value

			tup = (startna, lastna, interpolate_start, interpolate_end)

			interpolate_tuples.append(tup)

			startna = None
			lastna = None
			interpolate_start = np.nan
			interpolate_end = np.nan

	if np.isnan(value):
		lastna = index
		tup = (startna, lastna, interpolate_start, interpolate_end)
		interpolate_tuples.append(tup)

	indices = np.array([])
	values = np.array([])

	for i in range(len(interpolate_tuples)):

		start, end, start_val, end_val = interpolate_tuples[i]

		# The start of the chromosome is nas:
		if np.isnan(start_val):

			cur_indices = np.arange(start, end+step, step)
			cur_values = np.repeat(end_val, len(cur_indices))

		# The end of the chromosome is nas:
		elif np.isnan(end_val):

			cur_indices = np.arange(start, end+step, step)
			cur_values = np.repeat(start_val, len(cur_indices))

		# Some string of nas somewhere inside the chromosome that can be interpolated
		else:        

			cur_indices = np.arange(start, end+step, step)
			k = len(cur_indices)
			# Inteporlate values from the start to the end
			# the start and end values represent values outside of the nan string
			# so add two, then take in the inner k elements as the values we will keep
			cur_values = np.linspace(start_val, end_val, k+2)[1:-1]

			# Create a function that returns a list of integers that interpolate
			# from a given start and end index and start and end value

		indices = np.concatenate([indices, cur_indices])
		values = np.concatenate([values, cur_values])
	interpolated_df = pd.DataFrame({'index': indices, 
		'value': values}).astype(int).set_index('index')
	return interpolated_df

