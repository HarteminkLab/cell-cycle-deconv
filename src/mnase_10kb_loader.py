
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.chromatin_model import read_chromosome_mnase_reads
from src.TracerPlotter import normalize_max_min
from src.global_config import GlobalConstants


class MNase10kbLoader:
	"""Class to load 10kb occupancies
	"""

	def __init__(self):
		pass

	def load_mnase_data(self, replicate, chromosome, fragment_lengths_span=None):
		from src.sgd import get_chromosome_length

		self.replicate = replicate
		self.chromosome = chromosome
		self.mnase_reads = read_chromosome_mnase_reads(replicate, chromosome)

		# todo: Here  add the option to filter mnase reads by fragment length***
		if fragment_lengths_span is not None:

			# Filter by fragment lengths
			selection_criteria = (self.mnase_reads['length'] >= fragment_lengths_span[0]) & (self.mnase_reads['length'] < fragment_lengths_span[1])
			self.mnase_reads = self.mnase_reads[selection_criteria]

		self.timepoints = self.mnase_reads['sample'].unique()
		self.chrom_len = get_chromosome_length(self.chromosome)

		# Compute read counts
		chrom_read_counts = self.mnase_reads.groupby(['sample', 'mid']).count()
		self.chrom_read_counts = chrom_read_counts[['start']].rename(columns={'start': 'count'})

	def compute_sliding_window_counts_all_times(self, window_size=GlobalConstants.REPL_DECONV_BIN_WIDTH,
			step=GlobalConstants.REPL_DECONV_BIN_STEP, min_count_thresh = 0.7):
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
		self.all_counts_unnormalized_df = pd.DataFrame(self.all_counts_unnormalized.T,
			index=self.start_indices)
		self.all_counts_unnormalized_df.columns = timepoints

		# Normalize to equal samples
		unnormalized_total_occupancy = self.all_counts_unnormalized_df
		normalized_total_occ = unnormalized_total_occupancy / \
			unnormalized_total_occupancy.mean(axis=0).values.reshape((1, -1))
		self.normalized_total_occupancy_df = normalized_total_occ

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


def compute_sliding_window(data, window_size, step):
	# Number of windows
	n_windows = (len(data) - window_size) // step + 1
	
	# Initialize an array to hold the result
	result = np.empty((n_windows, window_size))
	
	for i in range(n_windows):
		start = i * step
		result[i, :] = data[start:start + window_size]
	
	return result


def get_bin_for_position(position, start_indices, window_size=10000, step=2000):
	"""Get the bin in which the position is the closest to the center of the bin"""

	win = window_size
	win_2 = win//2

	position_bin_idx = np.argmin((position - win_2) > start_indices)
	bin_start = position_bin_idx * step
	bin_end = bin_start + win

	return position_bin_idx, bin_start
