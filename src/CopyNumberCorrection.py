
import numpy as np
import matplotlib.pyplot as plt

class CopyNumberCorrection:
	
	def __init__(self, reads, replication_profile):

		self.reads = reads
		self.replication_profile = replication_profile
		self.corrected_reads = correct_copy_number(reads, replication_profile)


	def plot_observed_vs_corrected(self):
		plot_observed_vs_corrected(self.replication_profile, self.reads, self.corrected_reads)

		
def correct_copy_number(reads, replication_profile):
	"""
	Corrects the read counts by copy number based on the replication profile.

	Parameters:
	reads (2D array): Observed read counts at each time point.
	replication_profile (2D array): Replication profile indicating the copy number at each time point.

	Returns:
	2D array: Copy number corrected read counts.
	"""
	# Compute the average copies per genome at each time point
	# copies_per_genome = replication_profile.mean(axis=1)

	# Correct for actual replication status
	corrected_reads = reads / replication_profile

	# Re-normalize to ensure each time point sums to the same total
	total_reads_per_timepoint = reads.sum(axis=1).reshape((-1, 1))
	normalization_factors = total_reads_per_timepoint / corrected_reads.sum(axis=1).reshape((-1, 1))
	normalized_corrected_reads = corrected_reads * normalization_factors

	return normalized_corrected_reads


def plot_reads_bar(corrected_reads, scale=100, color='gray'):

	def plot_row(corrected_reads, row):
		dat = corrected_reads[row, :]
		dat = np.concatenate([dat[0:], dat[-1:]])
		xs = np.arange(len(dat))
		y_offset = np.zeros_like(dat) + row * 2*scale
		plt.fill_between(xs, dat+y_offset, y_offset, step='post', color=color)

	n = corrected_reads.shape[0]

	for i in range(n):
		plot_row(corrected_reads, i)

	yticks = np.arange(0, scale*n*2, scale)

	yticklabels = []
	for i in range(n):
		yticklabels = yticklabels + [0, scale]

	plt.yticks(yticks, yticklabels)


def plot_observed_vs_corrected(rep_profile, observed_reads, corrected_reads):
	plt.figure(figsize=(11, 6))
	plt.subplots_adjust(hspace=0.5)

	plt.subplot(2, 3, 1)
	plot_reads_bar(rep_profile, scale=2., color=plt.get_cmap('tab10')(0))
	plt.title("Replication profile")
	plt.xlabel("Genomic position")

	plt.subplot(2, 3, 2)
	plot_reads_bar(observed_reads)
	plt.title("Observed reads")
	plt.xlabel("Genomic position")

	plt.subplot(2, 3, 3)
	plot_reads_bar(corrected_reads)
	plt.title("Corrected reads")
	plt.xlabel("Genomic position")

	plt.subplot(2, 3, 4)
	plt.plot(rep_profile.mean(axis=1))
	plt.title("Average copies per genome")
	plt.xlabel("Time")

	plt.subplot(2, 3, 5)
	plt.plot(observed_reads.mean(axis=1))
	plt.title("Observed, total reads per time")
	plt.xlabel("Time")

	plt.subplot(2, 3, 6)
	plt.plot(corrected_reads.mean(axis=1))
	plt.title("Corrected, total reads per time")
	plt.xlabel("Time")
