
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.mnase_replication_timing_analysis import get_bin_for_position
from src.geneset import get_deconvolved_geneset
from src.helpers import calcH_config


class CopyNumberCorrection:
	"""Currently only used in toy example."""
	
	def __init__(self, reads, replication_profile):

		self.reads = reads
		self.replication_profile = replication_profile
		self.corrected_reads = correct_copy_number(reads, replication_profile)


	def plot_observed_vs_corrected(self):
		plot_observed_vs_corrected(self.replication_profile, self.reads, self.corrected_reads)

	
def get_gene_replication_profile(orf_name, repl_profile=None, geneset=None):
	"""Get the replication profile for a given orf. Note the profile returned
	using the Delta-DG1 model.
	
	Loads the deconvolved profile for the given gene orf, from the gene's 
	assigned bin
	
	Returns vector of the deconvolved profile [1 and 2's representing the copy number
	at each timepoint]
	"""

	if repl_profile is None:
		repl_profile = pd.read_csv('datasets/computed_mnase/'\
			'deconvolved_all_chr_replication_profile_delta_model.csv')
		repl_profile = repl_profile.set_index(['chr', 'start'])

	if geneset is None:
		from src.geneset import get_deconvolved_geneset
		geneset = get_deconvolved_geneset()

	gene = geneset.loc[orf_name]
	chrom = gene.chr
	start_indices = repl_profile.loc[chrom].index.values

	chrom_repl_profile = repl_profile.loc[chrom]
	bin_idx, bin_start_bp = get_bin_for_position(gene.TSS, start_indices)
	gene_repl_profil = chrom_repl_profile.loc[bin_start_bp]
	gene_repl_profil.index = gene_repl_profil.index.astype(int)
	
	return gene_repl_profil
	

def copy_number_correct(H, gene, data_to_correct, repl_profiles, geneset):
	"""Correct the gene data by copy number. Uses the H matrix and the replication timing
	for a gene to effectly reduce the proportion of data in copy number 2 timepoints to 
	1. 
	"""

	# Identify the time of replication
	replication_profile = get_gene_replication_profile(gene.name, repl_profiles, geneset)
	replication_idx = replication_profile.argmax()
	
	# Within the H matrix, identify the subset of
	# postG1 that will be copy number 2
	copy_2_H = H[:, replication_idx:-1]

	# Compute the proportion of copy number 2
	# and copy number 1 cells
	prop_cop2 = copy_2_H.sum(axis=1)
	prop_cop1 = 1 - prop_cop2
	
	# Correct the data by halving
	# the copy number 1 proportion
	data_cop1 = data_to_correct * prop_cop1
	data_cop2 = data_to_correct * prop_cop2 * 0.5
	
	# Recombine the copy number 1 and copy number 2 (corrected)
	data_corrected = data_cop1 + data_cop2

	return data_corrected
	

def correct_copy_number(reads, replication_profile):
	"""
	Used for the toy example.

	Corrects the read counts by copy number based on the replication profile. Both reads and replication profile
	should be in the same dimension (time along the y-axis and genomic-position/segment along the x-axis)

	Parameters:
		reads (2D array): Observed read counts at each time point.
		replication_profile (2D array): Replication profile indicating the copy number at each time point.

	Returns:
		2D array: Copy number corrected read counts.
	"""

	# Correct for the replication copy number at each genomic position for each timepoint
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

		y_offset = row * 2*scale
		y_offset_arr = np.zeros_like(dat)+y_offset

		plt.fill_between(xs, dat+y_offset, y_offset, step='post', color=color, lw=0)
		plt.axhline(y_offset, c='black', lw=0.75)

	n = corrected_reads.shape[0]
	m = corrected_reads.shape[1]

	for i in range(n):
		plot_row(corrected_reads, i)

	spacing_between_plots = scale*2
	yticks = np.arange(spacing_between_plots/2., spacing_between_plots*n, spacing_between_plots)
	yticklabels = ["${t_"+str(i+1)+"}$" for i in range(n)]

	plt.yticks(yticks, yticklabels)

	xticks = np.arange(0, m)
	xticklabels = xticks
	plt.ylim(-spacing_between_plots*0.25, n*spacing_between_plots)

	# plt.xticks(xticks+0.5, xticklabels)


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
