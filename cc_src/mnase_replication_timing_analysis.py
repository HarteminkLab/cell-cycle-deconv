
import numpy as np
import pandas as pd
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
		self.chrom_len = get_chromosome_length(self.chromosome)

		# Compute read counts
		chrom_read_counts = self.mnase_reads.groupby(['sample', 'mid']).count()
		self.chrom_read_counts = chrom_read_counts[['start']].rename(columns={'start': 'count'})


	def compute_sliding_window_counts_all_times(self, window_size=10000, step=5000):
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


def sliding_window_approach(data, window_size, step):
	# Number of windows
	n_windows = (len(data) - window_size) // step + 1
	
	# Initialize an array to hold the result
	result = np.empty((n_windows, window_size))
	
	for i in range(n_windows):
		start = i * step
		result[i, :] = data[start:start + window_size]
	
	return result
