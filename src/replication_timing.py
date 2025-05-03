
import pandas as pd

class ReplicationTiming:
	"""Class to handle loading and retrieving replication timing 
	for genomic features"""

	def __init__(self, output_dir):
		"""Load with the appropriate deconvolution run.

		We may cache this, if this takes to long to load every time
		"""
		self.output_dir = output_dir

		from src.RealDataReplication import load_replication_Fr_df

		repl_dfs = []
		for chrom in range(1, 17):
			_, chrom_replication_df = load_replication_Fr_df(output_dir, chrom, 
				with_replication_timing=True)
			chrom_replication_df['chr'] = chrom
			repl_dfs.append(chrom_replication_df)

		self.replications_df = pd.concat(repl_dfs).reset_index().set_index(['chr', 'start'])

	def compute_peak_annotations(self):
		"""Compute whether a replication timing occurs at a peak, that is it is the
		earlier than its neighbors. Peaks will be annotated as shared or absolute. For
		shared peaks, it will be equal to either neighbor."""
		
		chrom_peaks_dfs = []
		for chrom in range(1, 17):
			chrom_peaks = self.compute_peak_annotations_chrom(chrom)
			chrom_peaks['chr'] = chrom
			chrom_peaks_dfs.append(chrom_peaks)
		peaks_df = pd.concat(chrom_peaks_dfs).reset_index().set_index(['chr', 'start'])
		self.replications_df = self.replications_df.join(peaks_df)

	# Now let's identify indices with peaks
	def compute_peak_annotations_chrom(self, chrom):
		"""Compute whether a replication timing occurs at a peak, that is it is the
		earlier than its neighbors. Peaks will be annotated as shared or absolute. For
		shared peaks, it will be equal to either neighbor."""
		chrom_repl_timings = self.replications_df.loc[chrom]
		peak_annotations = chrom_repl_timings[[]].copy()
		peak_annotations['peak_type'] = False
		start_indices = chrom_repl_timings.index
		peaks = []
		for index, start in enumerate(start_indices):
			if index > 0 and index < len(start_indices)-1:
				prev_entry = chrom_repl_timings.iloc[index-1]
				cur_entry = chrom_repl_timings.iloc[index]
				next_entry = chrom_repl_timings.iloc[index+1]

				if (cur_entry.replication_time < prev_entry.replication_time and
					cur_entry.replication_time < next_entry.replication_time):
					peak_annotations.loc[start, 'peak_type'] = 'absolute'

				elif (cur_entry.replication_time <= prev_entry.replication_time and
					cur_entry.replication_time <= next_entry.replication_time):
					peak_annotations.loc[start, 'peak_type'] = 'shared'
		return peak_annotations

	def load_replication_entry_for(self, chrom, position=None, span=None):

		from src.mnase_10kb_loader import get_bin_for_position
		start_indices = self.replications_df.loc[chrom].index
		if span is not None:
			position = (span[0]+span[1])/2

		bin_idx, start = get_bin_for_position(position, start_indices)
		return self.replications_df.loc[chrom].loc[start]
