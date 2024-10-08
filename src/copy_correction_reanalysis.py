

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.figure_configs import FiguresConfig
from src.global_config import GlobalConstants
from src.utils import print_fl

class CopyCorrector:

	def __init__(self, genome_deconvolution):

		from src.config import load_configs_by_config_type
		from src.stepwise_replication_solver import load_chrom_replication_profile

		# Proposed new copy correction procedure, take the replication time and the 
		# proportion of cells in S/G2/M and divide by 2 for each corresponding timepoint...
		# Hoping it makes the data look less wavey
		config1, config2 = load_configs_by_config_type('shared')
		(H1, _), (H2,  _) = config1.calcH_function(config1.intervals_wt1, config1.WT1_TIMEPOINTS), \
			config2.calcH_function(config2.intervals_wt1, config2.WT1_TIMEPOINTS)

		replication_profile = load_chrom_replication_profile()

		self.genome_deconvolution = genome_deconvolution
		self.chrom_replication_profile = replication_profile
		self.config1 = config1
		self.config2 = config2
		self.H1, self.H2 = H1, H2

	def retrieve_replication_proportions(self, replicate):

		chrom, span = self.genome_deconvolution.combined_model.chrom1_model.chr, \
			self.genome_deconvolution.combined_model.chrom1_model.mnase_span

		H = self.H1 if replicate == 1 else self.H2
		replication_idx = int(self.chrom_replication_profile.loc[chrom].loc[span[0]].replication_index)
		end_g2m_idx = H.shape[1]-2

		# Total uncorrected Sum from H
		total_uncorrected_sum = H.sum(axis=1)

		# Compute the replication proportion of reads from the replication index through
		# the end of G2M
		replicated_proportion = np.zeros_like(H)

		replicated_proportion[:, replication_idx:end_g2m_idx] = H[:, replication_idx:end_g2m_idx]
		replicated_sum = replicated_proportion.sum(axis=1)
		proportion_of_replicated_DNA = replicated_sum / (total_uncorrected_sum + replicated_sum)

		self.replication_idx = replication_idx
		self.replicated_proportion = replicated_proportion
		self.proportion_of_replicated_DNA = proportion_of_replicated_DNA


	def perform_correction(self, replicate):

		if replicate == 1:
			normalized_bins = self.genome_deconvolution.combined_model.chrom1_model.normalized_bins
		else:
			normalized_bins = self.genome_deconvolution.combined_model.chrom2_model.normalized_bins

		uncorrected_reads = normalized_bins.reshape((normalized_bins.shape[0], -1))
		uncorrected_reads_sum = uncorrected_reads.sum(axis=1)
		self.uncorrected_reads_sum = uncorrected_reads_sum

		try:
			self.retrieve_replication_proportions(replicate)
		except KeyError:
			chrom, span = self.genome_deconvolution.combined_model.chrom1_model.chr, \
				self.genome_deconvolution.combined_model.chrom1_model.mnase_span
			print_fl(f"Could not retrieve replication profile for: {chrom}, {span}")
			print_fl("Applying no correction")

			self.corrected_reads = uncorrected_reads
			self.corrected_reads_sum = self.uncorrected_reads_sum
			self.correction_scalar = np.ones_like(self.corrected_reads)
			return 

		replication_idx, replicated_proportion, \
			proportion_of_replicated_DNA = self.replication_idx, self.replicated_proportion, \
			self.proportion_of_replicated_DNA

		# Compute the correction scalar using the sum of the window
		corrected_reads_sum = uncorrected_reads_sum - uncorrected_reads_sum*proportion_of_replicated_DNA
		correction_scalar = corrected_reads_sum / uncorrected_reads_sum

		# Perform the correction against the original unsummed matrix
		corrected_reads = uncorrected_reads * correction_scalar.reshape((-1, 1))

		# Following correction, the overall number of reads will drop compared to the
		# original sum, so this will need to be adjusted genome-wide
		self.corrected_reads = corrected_reads
		self.corrected_reads_sum = corrected_reads_sum
		self.correction_scalar = correction_scalar


	def plot_correction_H(self):

		H1 = self.H1

		# Then, from replication until the end of G2/M, we will take this proportion of 
		# reads and divide them by 2...

		plt.figure(figsize=(12, 3.5))
		plt.subplots_adjust(bottom=0.2)
		plt.subplot(1, 3, 1)
		plt.imshow(H1, vmax=0.02, aspect='auto', 
			interpolation='none')
		plt.title("$\\bf{H}$")
		plt.xticks([])
		plt.yticks([])
		plt.xlabel("Single cell time")
		plt.ylabel("Experimental time")

		plt.subplot(1, 3, 2)

		self.retrieve_replication_proportions(1)

		replication_idx, replicated_proportion, \
			proportion_of_replicated_DNA = self.replication_idx, self.replicated_proportion, \
			self.proportion_of_replicated_DNA
		end_g2m_idx = H1.shape[1]-2

		plt.imshow(H1+replicated_proportion, vmax=0.02, aspect='auto', 
			interpolation='none')
		plt.title("$\\bf{H}$ with replicated DNA")
		plt.xticks([(replication_idx+end_g2m_idx)/2], ["Replicated DNA"])
		plt.yticks([])
		plt.axvline(replication_idx, c='red', ls='solid')
		plt.axvline(end_g2m_idx, c='red', ls='solid')
		plt.gca().xaxis.set_tick_params(pad=5, length=0)

		# We assume that the replicated proportion of DNA is included in the
		# total number of reads, so normalize by the total proportion and the additional
		# copy corrected DNA
		total_uncorrected_sum = H1.sum(axis=1)
		replicated_sum = replicated_proportion.sum(axis=1)

		plt.subplot(1, 3, 3)
		timepoints = self.config1.WT1_TIMEPOINTS
		plt.plot(timepoints, total_uncorrected_sum, label="Total proportion of DNA",
			color=plt.get_cmap('tab10')(0))
		plt.plot(timepoints, total_uncorrected_sum+replicated_sum, label="Additionally replicated DNA",
			color=plt.get_cmap('tab10')(1))
		plt.plot(timepoints, total_uncorrected_sum-replicated_sum, label="Proportion with\nreplicated DNA removed",
			color=plt.get_cmap('tab10')(0), ls='dotted')
		plt.title("Copy correction proportion")
		plt.legend()
		plt.yticks(np.arange(0, 2, 0.5))
		plt.ylim(-0.5, 1.75)
		plt.xlabel("Experimental time")



def load_chromatin_copy_correction(config_type, replicate):
	path = f'data/copy_correction/chromatin/copy_correction_replicate{replicate}.csv'
	correction = pd.read_csv(path).set_index(['chr', 'start'])
	return correction


def lookup_origin_copy_correction(copy_correction, origin):
	"""Look up the copy correction vector for an origin from the
	genomic copy correction table"""
	
	from src.CopyNumberCorrection import get_bin_for_position

	chrom = origin.chr
	pos = gene.pos

	start_indices = copy_correction.loc[chrom].index
	bin_idx, bin_start_bp = get_bin_for_position(pos, start_indices)

	vector = copy_correction.loc[chrom].loc[bin_start_bp]
	return vector


def lookup_copy_correction(copy_correction, chrom, pos):
	"""Look up the copy correction vector for the gene from the
	genomic copy correction table"""

	from src.CopyNumberCorrection import get_bin_for_position

	start_indices = copy_correction.loc[chrom].index
	bin_idx, bin_start_bp = get_bin_for_position(pos, start_indices)

	vector = copy_correction.loc[chrom].loc[bin_start_bp].values

	# todo: if there is an issue with the 10k window copy correction
	# the correction vector defaults to 0's. Rather we would like to 
	# interpolate from neighbors
	if vector.sum() == 0:
		print("Copy correction vector does not exist, using no correction. "
			  f"position: {pos}, chr: {chrom}, {pos}")
		vector = np.ones_like(vector)

	return vector


def lookup_gene_copy_correction(copy_correction, gene):
	"""Look up the copy correction vector for the gene from the
	genomic copy correction table"""

	from src.CopyNumberCorrection import get_bin_for_position

	chrom, pos = gene.chr, gene.TSS
	start_indices = copy_correction.loc[chrom].index
	bin_idx, bin_start_bp = get_bin_for_position(gene.TSS, start_indices)

	vector = copy_correction.loc[chrom].loc[bin_start_bp].values

	# todo: if there is an issue with the 10k window copy correction
	# the correction vector defaults to 0's. Rather we would like to 
	# interpolate from neighbors
	if vector.sum() == 0:
		print("Copy correction vector does not exist, using no correction. "
			  f"gene: {gene['gene']}, chr: {chrom}, {pos}")
		vector = np.ones_like(vector)

	return vector


def perform_precomputed_correction_normalisation(bins, chrom, span, replicate):
	"""Load the precomputed correction and normalization scalar from disk.
	This was precomputed because the normalization must be done across the 
	entire genome following copy correction. (Copy correction reduces the
	number of reads for the latter timepoints)"""

	filename = f'data/copy_correction/chromatin/copy_correction_replicate{replicate}.csv'
	correction_normalization_scalar = pd.read_csv(filename).set_index(['chr', 'start'])
	correction_norm_scalar = correction_normalization_scalar.loc[chrom].loc[span[0]]
	corrected_normalized_bins = bins * correction_norm_scalar.values.reshape((-1, 1, 1))

	return corrected_normalized_bins

