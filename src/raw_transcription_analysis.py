
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from src.read_bam import get_rna_seq_filepaths_df
# from src.read_bam import read_rna_bam
# from src.sgd import get_chromosome_length



# Deprecated, unsure of the goals and findings of this analysis.
# I don't believe RNA-seq was found to be related to replication timing.



# class TranscriptionDataAnalysis:
# 	"""Class to analyze gene transcription chromosomes
# 	and compare them to the origin of replication timing in the Brewer lab

# 	1. Load the rna-seq reads for a chromsome
# 	2. Normalize by number of reads per timepoint/sample 
# 	3. Normalize by the average value for the timepoint.
# 	3. Bin the reads by 10k windows
# 	4. Create a heatmap of the reads.
# 	"""

# 	def __init__(self, window=2000):
# 		self.window = window

# 	def load_replicate_chromosome(self, replicate, chrom):

# 		bam_df = get_rna_seq_filepaths_df()
# 		bam_df = bam_df[bam_df.replicate == replicate]

# 		self.chrom = chrom
# 		self.replicate = replicate

# 		chr_reads = pd.DataFrame()
# 		for idx, row in bam_df.iterrows():
# 			time = row.time
# 			filepath = row.full_path
# 			rna_reads = read_rna_bam(filename=filepath, chroms=[chrom],
# 				sample=time)
# 			chr_reads = pd.concat([chr_reads, rna_reads])
# 		self.chr_reads = chr_reads
# 		self.timepoints = self.chr_reads['sample'].unique()


# 	def calculate_bins(self):

# 		from src.sgd import get_chromosome_length

# 		chr_reads = self.chr_reads
# 		chrom_length = get_chromosome_length(self.chrom)

# 		m = len(self.timepoints)

# 		bins = np.arange(0, chrom_length, self.window)

# 		chr_window_counts = np.zeros((m, len(bins)-1))

# 		for i in range(m):
# 			timepoint = self.timepoints[i]
# 			timepoint_reads = chr_reads[chr_reads['sample'] == timepoint]
# 			rna_hist, _ = np.histogram(timepoint_reads['start'], bins=bins)

# 			chr_window_counts[i] = rna_hist

# 		# Normalize such that each timepoint is equal
# 		chr_window_counts_normalized_by_time = chr_window_counts.copy()

# 		timepoint_sums = chr_window_counts.sum(axis=1)

# 		for i in range(m):
# 			chr_window_counts_normalized_by_time[i] = chr_window_counts[i]\
# 				/ timepoint_sums[i] *10000.

# 		chr_window_counts_normalized_by_time.sum(axis=1)

# 		# Now normalize such that each vertical 10k window are relative to the mean'
# 		window_means = chr_window_counts_normalized_by_time.mean(axis=0)
# 		chr_counts_window_normalized = chr_window_counts_normalized_by_time.copy()

# 		for i in range(chr_counts_window_normalized.shape[1]):
# 			chr_counts_window_normalized[:, i] = chr_window_counts_normalized_by_time[:, i] \
# 				/ window_means[i]
				
# 		self.log_normalized_counts = np.log2(chr_counts_window_normalized)
# 		self.chr_counts_window_normalized = chr_counts_window_normalized
# 		self.chr_window_counts_normalized_by_time = chr_window_counts_normalized_by_time


# 	def plot_heatmap(self):
# 		fig = plt.figure(figsize=(18, 9))
# 		chrom_length = get_chromosome_length(self.chrom)
# 		m = len(self.timepoints)

# 		plt.imshow(self.log_normalized_counts, origin='lower', cmap='RdBu_r', 
# 				   extent=[0, chrom_length, 0,
# 			m], aspect='auto', vmin=-2, vmax=2)
# 		plt.yticks(np.arange(m)+0.5, self.timepoints)
# 		plt.colorbar()
# 		plt.title(f"Chromosome {self.chrom}\nwindow={self.window}, "
# 				  f"replicate={self.replicate}", 
# 				  fontsize=26, pad=20)
# 		return fig
