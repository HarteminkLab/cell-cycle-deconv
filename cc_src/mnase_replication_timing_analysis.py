
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cc_src.chromatin_model import read_chromosome_mnase_reads


class MNaseOriginAnalysis:
	"""Analysis to determine correlation of cell cycling replication timing with
	MNase-seq read counts using sliding windows"""

	def __init__(self):
		pass

	def load_mnase_data(self, replicate, chromosome):
		from cc_src.sgd import get_chromosome_length

		self.replicate = replicate
		self.chromosome = chromosome
		self.mnase_reads = read_chromosome_mnase_reads(replicate, chromosome)
		self.timepoints = self.mnase_reads['sample'].unique()

		print(self.timepoints)

		self.chrom_len = get_chromosome_length(self.chromosome)

		# Compute read counts
		chrom_read_counts = self.mnase_reads.groupby(['sample', 'mid']).count()
		self.chrom_read_counts = chrom_read_counts[['start']].rename(columns={'start': 'count'})


	def compute_sliding_window_counts_all_times(self, window_size=10000, step=2000):
		"""Compute the sliding window counts"""

		self.window_size = window_size
		self.step = step

		timepoints = self.timepoints
		num_windows = (self.chrom_len - self.window_size) // self.step + 1

		all_window_counts = np.zeros((len(timepoints), num_windows))

		i = 0
		for timepoint in timepoints:
			
			self.compute_sliding_window_counts(timepoint, window_size, step)
			all_window_counts[i, :] = self.window_counts.sum(axis=1)
			i += 1

		self.all_window_counts = all_window_counts


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
		self.window_counts = sliding_window_approach(counts, window_size, step)

	def normalize_samples(self):
		"""Normalize by expected DNA content at each time point"""
		counts_normalized_by_copy = self.all_window_counts.copy()
		counts_normalized_equal = self.all_window_counts.copy()

		if self.replicate == 2:

			# Computed from the 00_Plot_H_Growth_Curves notebook
			# Estimates the sample size by expected occupancy count for G1, S, G2M at each timepoint

			# Normalized with mass scaling and dna content
			# norm_scale_vals = np.array([1.        , 1.05373696, 1.17750614, 1.35511446, 1.54620776,
			# 	   1.73200968, 1.88853892, 2.05493022, 2.30677725, 2.64861854,
			# 	   3.0239524 , 3.38152438, 3.72040055, 4.09746831, 4.57313829])


			# Normalized without mass
			norm_scale_vals = np.array([1.        , 1.05373365, 1.1772379 , 1.34687222, 1.45824708,
			       1.38450379, 1.21878375, 1.14307137, 1.19364682, 1.31193423,
			       1.39543665, 1.38009267, 1.31063579, 1.26561563, 1.2737374 ])


		else:
			raise ValueError("Unimplemented.")

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

		plt.imshow(data, aspect='auto', origin='lower',
				  extent=[0, self.chrom_len, -5, timepoints[-1]+5])

		plt.yticks(timepoints)

		plt.axhline(self.g1_recovery_would_start_here, c='white', alpha=0.125)
		plt.axhline(self.first_s_start, c='white', alpha=0.125)
		plt.axhline(self.first_s_end, c='white', alpha=0.125)
		plt.axhline(self.end_of_first_lambd, c='white', alpha=0.125)
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

		for bin_num in selected_bins:
			plt.plot(self.timepoints, data[:, bin_num], 
				label=f"Bin {bin_num*self.step//1000}K")

		plt.axvline(self.end_of_first_lambd, c='black', lw=1,
					label="End of first cell cycle", zorder=0)
		plt.axhline(2, c='black', lw=1, ls='dotted', zorder=0)
		plt.axhline(1, c='black', lw=1, ls='dotted', zorder=0)
		plt.legend()

		plt.xlabel("Occupancy")
		plt.ylabel("Time, minutes")

		plt.title("Bin occupancy over time", pad=10)


def sliding_window_approach(data, window_size, step):
	# Number of windows
	n_windows = (len(data) - window_size) // step + 1
	
	# Initialize an array to hold the result
	result = np.empty((n_windows, window_size))
	
	for i in range(n_windows):
		start = i * step
		result[i, :] = data[start:start + window_size]
	
	return result
