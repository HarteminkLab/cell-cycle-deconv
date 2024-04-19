
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.chromatin_model import read_chromosome_mnase_reads


class MNaseOriginAnalysis:
	"""Analysis to determine correlation of cell cycling replication timing with
	MNase-seq read counts using sliding windows"""

	def __init__(self):
		pass

	def load_mnase_data(self, replicate, chromosome):
		from src.sgd import get_chromosome_length

		self.replicate = replicate
		self.chromosome = chromosome
		self.mnase_reads = read_chromosome_mnase_reads(replicate, chromosome)
		self.timepoints = self.mnase_reads['sample'].unique()
		self.chrom_len = get_chromosome_length(self.chromosome)

		# Compute read counts
		chrom_read_counts = self.mnase_reads.groupby(['sample', 'mid']).count()
		self.chrom_read_counts = chrom_read_counts[['start']].rename(columns={'start': 'count'})


	def compute_sliding_window_counts_all_times(self, window_size=10000, step=2000,
			min_count_thresh = 0.7):
		"""Compute the sliding window counts"""

		self.window_size = window_size
		self.step = step
		self.min_count_thresh = min_count_thresh

		timepoints = self.timepoints
		num_windows = (self.chrom_len - self.window_size) // self.step + 1

		self.all_window_counts_unsummed = np.zeros((len(timepoints), num_windows, window_size))

		i = 0
		for timepoint in timepoints:
			
			window_counts = self.compute_sliding_window_counts(timepoint, window_size, step)
			self.all_window_counts_unsummed[i] = window_counts
			i += 1

		cumulative_counts_for_each_10k_bin = self.all_window_counts_unsummed.sum(axis=0)
		self.cumulative_counts_for_each_10k_bin = cumulative_counts_for_each_10k_bin

		all_counts_summed = self.all_window_counts_unsummed.sum(axis=2)
		self.all_counts_unnormalized = all_counts_summed

		# Normalize by the number of bins that are non-zero across the entire timeecourse
		self.total_nonzero_bins_per_10k = (cumulative_counts_for_each_10k_bin > 0).sum(axis=1)
		self.all_window_counts_nonzero_handled = all_counts_summed / self.total_nonzero_bins_per_10k

		# Multiply against mask to make sure the counts meet the minimum number of 
		# non-zero bins
		self.meets_threshold = self.total_nonzero_bins_per_10k > min_count_thresh*self.window_size
		self.all_window_counts = self.all_window_counts_nonzero_handled * self.meets_threshold

	def plot_bin_normalization_procedure(self):
		plt.figure(figsize=(19, 13))
		plt.subplots_adjust(hspace=0.5)
		plt.subplot(4, 1, 1)
		plt.imshow(self.all_window_counts_nonzero_unhandled, aspect='auto')
		plt.xticks([])
		plt.yticks([])
		plt.title("Unnormalized",
				  fontsize=22, pad=10)

		plt.subplot(4, 1, 2)
		plt.imshow(self.all_window_counts_nonzero_handled, aspect='auto')
		plt.xticks([])
		plt.yticks([])
		plt.title("Normalized by non-zero bin counts",
				  fontsize=22, pad=10)

		plt.subplot(4, 1, 3)
		plt.imshow(self.all_window_counts, aspect='auto')
		plt.xticks([])
		plt.yticks([])
		plt.title(f"Meets threshold, threshold={self.min_count_thresh}",
				  fontsize=22, pad=10)

		# Identify a cutoff in which we should zero out the count curves
		plt.subplot(4, 1, 4)
		xs = self.start_indices + self.window_size//2
		plt.plot(xs, self.total_nonzero_bins_per_10k,
			label="Non-zero bins")
		plt.plot(xs, 
			self.total_nonzero_bins_per_10k * self.meets_threshold,
				label="Non-zero bins * meets threshold")
		plt.xlim(0, self.chrom_len)
		plt.axhline(self.min_count_thresh * self.window_size, c='black',
				   lw=1, ls='dotted', label='threshold')
		plt.title(f"Curves of bin counts and threshold multiplier, threshold={self.min_count_thresh}",
				  fontsize=22, pad=10)
		plt.legend()
		plt.suptitle(f"Non-zero bin normalization, Replicate {self.replicate},"
					 f" chr{self.chromosome}", fontsize=29)


	def get_bin_for_position(self, position):
		"""Get the bin in which the position is the closest to the center of the bin"""

		start_indices = self.start_indices
		win = self.window_size
		win_2 = win//2
		step = self.step

		position_bin_idx = np.argmin((position - win_2) > start_indices)
		bin_start = position_bin_idx * step
		bin_end = bin_start + win

		return position_bin_idx, bin_start

	def compute_sliding_window_counts(self, time, window_size, step):
		"""Compute the sliding window counts"""

		chrom_len = self.chrom_len
		counts = self.chrom_read_counts.loc[time]['count']
		new_index = pd.RangeIndex(start=0, stop=chrom_len)
		reindexed_series = counts.reindex(new_index)
		counts = reindexed_series.fillna(0)

		shape = ((counts.size - window_size) // step + 1, window_size)

		# Generate the start indices for each window
		self.num_windows = (counts.size - window_size) // step + 1
		self.start_indices = np.arange(self.num_windows) * step
		window_counts = compute_sliding_window(counts, window_size, step)
		return window_counts

	def normalize_samples(self):
		"""Normalize by expected DNA content at each time point"""
		counts_normalized_by_copy = self.all_window_counts.copy()
		counts_normalized_equal = self.all_window_counts.copy()

		norm_scale_vals = pd.read_csv(f'datasets/computed_mnase/dna_copy_scaling_rep{self.replicate}.csv')
		norm_scale_vals = norm_scale_vals['scale'].values

		for i in range(len(norm_scale_vals)):

			# Divide by sum, multiply by scalar
			# And by the number of window (this causes each bin to transition from 1 to 2 in the first
			# cell cycle)
			counts_normalized_by_copy[i] *= 1./counts_normalized_by_copy[i].sum() * norm_scale_vals[i] * \
				self.num_windows
			counts_normalized_equal[i] *= 1./counts_normalized_equal[i].sum() * self.num_windows

		self.counts_normalized_by_copy = counts_normalized_by_copy
		self.counts_normalized_equal = counts_normalized_equal


	def load_posteriors_for_timing(self):
		# Next, estimate when S starts and ends, probably 30-40 minutes
		# Take the heatmap of these timepoints and estimate when replication occurred along the 
		# chromosome to estimate the early/late activation origin proximity.

		from src.create_models import read_cloccs_posteriors
		params = read_cloccs_posteriors(f'data/2019_cloccs_fits/yl_2019_replicate{self.replicate}/posteriors.txt')

		mu0, lambda_len, gamma1, gamma2 = params['mu0'], params['lambda'], params['gamma1'], \
			params['gamma2']

		if self.replicate == 2:
			alpha = 22
		else:
			alpha = 28

		# Estimate the first S from mu0, lambda, gamma1, and gamma2
		cg1_length = gamma1*lambda_len+alpha
		s_start = (lambda_len*gamma1)
		s_end = (lambda_len*gamma2)
		s_length = s_end - s_start

		# For the first cell cycle, mu0 includes the first G1
		# so S starts when Recovery (mu0) ends
		self.first_s_start = -mu0
		self.first_s_end = -mu0+s_length

		# The end of the first cycle is computed
		# by taking the cell cycle length, subtracting the length of S (to get G1 and G2/M)
		# Then subtract out what the first G1 would be.
		# Then offset by mu0 length to get the actual timepoint for the end of the first cell cycle
		self.g1_recovery_would_start_here = self.first_s_start - cg1_length
		self.end_of_first_lambd = self.g1_recovery_would_start_here+lambda_len

		self.params = params


	def plot_heatmap(self, plot_norm_by_copy=True):
		"""Plot the window bins for the chromosome normalized or unnormalized"""

		self.load_posteriors_for_timing()

		timepoints = self.timepoints

		plt.figure(figsize=(16, 9))


		if plot_norm_by_copy:
			data = self.counts_normalized_by_copy
		else:
			data = self.counts_normalized_equal

		plt.imshow(data, aspect='auto', origin='lower', cmap='magma',
				  extent=[0, self.chrom_len, -5, timepoints[-1]+5], vmax=2., vmin=1.)
		plt.colorbar()

		plt.yticks(timepoints)

		plt.axhline(self.g1_recovery_would_start_here, c='white', alpha=0.5, lw=2, ls='dotted')
		plt.axhline(self.first_s_start, c='white', alpha=0.5, lw=2, ls='dotted')
		plt.axhline(self.first_s_end, c='white', alpha=0.5, lw=2, ls='dotted')
		plt.axhline(self.end_of_first_lambd, c='white', alpha=0.5, lw=2, ls='dotted')
		plt.title(f"Replicate {self.replicate}, chr{self.chromosome}, mnase-seq normalized samples", 
			fontsize=24, pad=15)
		plt.xlabel("Genomic position, nt", fontsize=16)
		plt.ylabel("Time, minutes", fontsize=16)

	def plot_bin_occupancy_for_selected_timepoints(self, 
		time_indices, selected_bins, plot_norm_by_copy=True):

		timepoints = self.timepoints
		plt.figure(figsize=(13, 3))

		cmap = plt.get_cmap('plasma_r')
		
		k = len(time_indices)
		max_color_scale = 0.7
		min_color_scale = 0.3

		if plot_norm_by_copy:
			data = self.counts_normalized_by_copy
		else:
			data = self.counts_normalized_equal

		for i in range(k):
			index = time_indices[i]
			timepoint = timepoints[index]
			plot_data = data[index]
			
			# Select a subset of the plasma color scale
			color = cmap(i/k*max_color_scale+min_color_scale) 
			plt.plot(self.start_indices, plot_data, 
					 color=color,
					label=f"{timepoint} min")

		plt.xlim(0, self.chrom_len)
		plt.ylim(-1.25, 2.5)

		for cur_bin in selected_bins:
			bin_nt = cur_bin*self.step
			plt.axvline(bin_nt, c='black', alpha=0.25, zorder=0,
					   label=f"Bin {bin_nt//1000}K")

		plt.xlabel("Genomic position, nt")
		plt.ylabel("Occupancy")
		
		plt.legend(ncol=2, loc='lower right')
		plt.title(f"Replicate {self.replicate}, "
				  f"Chr{self.chromosome}, MNase-seq occupancy")

	def plot_selected_bin_occupancy(self, selected_bins, plot_norm_by_copy=True):
		fig = plt.figure(figsize=(6, 3))

		if plot_norm_by_copy:
			data = self.counts_normalized_by_copy
		else:
			data = self.counts_normalized_equal

		cmap = plt.get_cmap('inferno_r')
		i = 0
		for bin_num in selected_bins:
			color = cmap(i/len(selected_bins)*0.75 + 0.25)
			plt.plot(self.timepoints, data[:, bin_num], 
				label=f"Bin {bin_num*self.step//1000}K", color=color)
			i += 1

		plt.axvline(self.end_of_first_lambd, c='black', linestyle='dashed', lw=1,
					label="End of first cell cycle", zorder=0)
		plt.axhline(1, c='black', lw=1, ls='dotted')
		plt.axhline(2, c='black', lw=1, ls='dotted')

		plt.legend()

		plt.xlabel("Occupancy")
		plt.ylabel("Time, minutes")
		plt.ylim(0.8, 2.25)

		plt.title("Bin occupancy over time", pad=10)


	def get_replication_timing(self, thresh_prop=0.75, sel_timepoints=None, plot=True):

		occupancy_counts = self.counts_normalized_by_copy.copy()
		timepoints = self.timepoints

		# Select subset of the rows for the timepoints we are interesed in
		if sel_timepoints is not None:
			sel_indices = np.nonzero(np.isin(timepoints, sel_timepoints))[0]
			occupancy_counts = occupancy_counts[sel_indices, :]
			timepoints = sel_timepoints

		# Threshold for determining when a bin has replicated
		min_values = occupancy_counts.min(axis=0)
		max_values = occupancy_counts.max(axis=0)

		# When the occupancy of a the bin reaches the threshold
		# we will consider the location replicated
		thresholds = min_values + (max_values - min_values) * thresh_prop

		exceeds_threshold = occupancy_counts > thresholds
		replication_timing_idx = np.argmax(exceeds_threshold, axis=0)

		replication_timepoints = timepoints[replication_timing_idx]
		return replication_timepoints

	def compute_bin_curves_and_expression_correlations(self, replicate):

		from src.config import read_yl_vst_data_rep
		from scipy.stats import pearsonr
		from src.reference_data import load_analysis_genes
		from src.timer import Timer

		geneset = load_analysis_genes()

		self.load_mnase_data(replicate, 1)

		expression_vst_data = read_yl_vst_data_rep(replicate=replicate)

		# In the case of replicate 2, the gene expression has one fewer
		# timepoint, so select the columns for those timepoints in the
		# chromatin data
		curve_timepoints = expression_vst_data.columns
		bin_curves = expression_vst_data.copy()
		bin_curves.loc[:] = 0.

		sel_tps_mask = [True if t in curve_timepoints else False for t in self.timepoints]

		timer = Timer()

		for chrom in np.arange(1, 17):
			print(f"{chrom}", end=", ")
			chrom_genes = geneset[geneset.chr == chrom]

			self.load_mnase_data(replicate, chrom)
			self.compute_sliding_window_counts_all_times()
			self.normalize_samples()

			for orf_name, gene in chrom_genes.iterrows():
				bin_idx, bin_start = self.get_bin_for_position(gene.TSS)
				bin_copy_num_curve = self.counts_normalized_by_copy[sel_tps_mask, bin_idx]
				bin_curves.loc[orf_name] = bin_copy_num_curve

		expression_bin_correlation = expression_vst_data.T.corrwith(bin_curves.T)
		expression_bin_correlations = expression_bin_correlation.dropna()

		self.bin_curves = bin_curves
		self.expression_vst_data = expression_vst_data
		self.expression_bin_correlations = expression_bin_correlations

		timer.print_time()


def compute_sliding_window(data, window_size, step):
	# Number of windows
	n_windows = (len(data) - window_size) // step + 1
	
	# Initialize an array to hold the result
	result = np.empty((n_windows, window_size))
	
	for i in range(n_windows):
		start = i * step
		result[i, :] = data[start:start + window_size]
	
	return result
