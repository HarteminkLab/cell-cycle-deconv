
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.figure_configs import FiguresConfig


class CopyCorrectionAnalysis:
	"""In this class, copy correction is revisited. The idea is to simplify the copy
	number correction process and clarify step-by-step how the process is performed:

	Procedure:
	1. Retrieve the 10 kb occupancy counts for all chromosomes
	2. Normalize these occupancy counts to match up the first cell cycle min and max 
		occupancy values. 
		Assumption: All segments of the genome will reach these same minimum and maximal values
			Reality: may not be the case as cells begin entering G1 by late S/G2M, 
			the maximal value is lowered and mixed with G1 cells.
	3. Retrieve each of these 10 kb occupancy values for each gene.
	4. Sort by replication timing
	5. Retrieve the estimated replication timing index from the deconvolution run.
	4. Compute an estimated copy number for each timepoint for each gene. This is computed
		from H (CLOCCS predictions of mixture of G1,S,G2/M at each timepoint with granularity
		for replication timing index to indicate when replication occurs).
	5. Correct for copy number. Assuming normalization and scaling is sufficient, compute the
		copy corrected occupancy counts.

	Expectation:
	1. Occupancy values that peak during S phase should now be reduced such that the
		approximate changes in occupancy should be around 1.0.

	Notes to address:
	- Mind the normalization schemes for the copy number proportions computed from H
	- The normalization for the 10 kb occupancy windows:
		computed as min and maximal values in the first cell cycle time frame
	"""
	def __init__(self):

		from src.mnase_replication_timing_analysis import MNaseOriginAnalysis

		# Compute the 10kb occupancies per the genome
		mnase_occupancies = MNaseOriginAnalysis(replicate=1)
		mnase_occupancies.compute_bin_curves()
		mnase_occupancies.normalize_and_compute_raw_replication_timing()
		self.mnase_occupancies_1 = mnase_occupancies

		mnase_occupancies = MNaseOriginAnalysis(replicate=2)
		mnase_occupancies.compute_bin_curves()
		mnase_occupancies.normalize_and_compute_raw_replication_timing()
		self.mnase_occupancies_2 = mnase_occupancies

		# Compute H for replicate 1 and 2
		from src.config import load_configs_by_config_type
		config1, config2 = load_configs_by_config_type('shared')
		self.H1, Hpos = config1.calcH_function(config1.intervals_wt1, config1.WT1_TIMEPOINTS)
		self.H2, Hpos = config2.calcH_function(config2.intervals_wt1, config2.WT1_TIMEPOINTS)

		from src.geneset import get_deconvolved_geneset
		self.genes = get_deconvolved_geneset()

	def compute_gene_10k_counts(self, replicate):

		self.replicate = replicate

		from src.CopyNumberCorrection import get_bin_for_position

		genes = self.genes
		gene_positions = genes[[]].copy()
		gene_chr_counts = genes[[]].copy()

		if replicate == 1:
			chr_bin_curves = self.mnase_occupancies_1.normalized_bin_curves
		else:
			chr_bin_curves = self.mnase_occupancies_2.normalized_bin_curves

		for chrom in range(1, 17):

			chr_genes = genes[genes.chr == chrom]
			start_indices = chr_bin_curves.loc[chrom].index

			gene_positions['chr'] = chrom

			for orf_name, gene in chr_genes.iterrows():
				bin_idx, bin_start_bp = get_bin_for_position(gene.TSS, start_indices)
				gene_positions.loc[orf_name, 'bin_start'] = bin_start_bp
				values = chr_bin_curves.loc[chrom].loc[bin_start_bp].values
				gene_chr_counts.loc[orf_name, np.arange(len(values))] = values

		gene_positions.bin_start = gene_positions.bin_start.astype(int)

		genes_repl_profile = pd.read_csv('data/replication_timing/yl_2019/genes_replication_timing_shared.csv')
		genes_repl_profile = genes_repl_profile.set_index('orf_name')
		genes_repl_profile = genes_repl_profile.sort_values('replication_time')

		# Sort the gene counts by the sorted replication time
		gene_chr_counts = gene_chr_counts.loc[genes_repl_profile.index]

		self.gene_positions = gene_positions
		self.gene_10k_counts = gene_chr_counts
		self.genes_repl_profile = genes_repl_profile


	def compute_correction_matrix(self):
		from src.copy_correction_reanalysis import correct_replication_indices

		replication_indices = self.genes_repl_profile.replication_H_index.values.astype(int)
		from src.copy_correction_reanalysis import create_copy_number_H, correct_replication_indices

		H = self.H1 if self.replicate == 1 else self.H2

		(H_combined_copy_num, 
		 H_expected_copy_per_gene, 
		 H_normalized_copy_per_gene) = correct_replication_indices(H, replication_indices)

		self.H_combined_copy_num = H_combined_copy_num
		self.H_expected_copy_per_gene = H_expected_copy_per_gene 
		self.H_normalized_copy_per_gene = H_normalized_copy_per_gene
		self.H_expected_copy_number_sum = self.H_expected_copy_per_gene.sum(axis=2)

		# scale is an approximation based on copy number curves from H
		# due to halted cells
		scale = 0.7
		self.corrected_counts = (self.gene_10k_counts*scale+1) / self.H_expected_copy_number_sum

	def plot_heatmap_correction(self):
		from src.global_config import GlobalConstants

		gene_chr_counts = self.gene_10k_counts

		# scale is an approximation based on copy number curves from H
		# due to halted cells
		scale = .7
		offset = 1

		plt.figure(figsize=(6, 6))
		# plt.subplot(1, 2, 1)
		# plt.imshow(self.H_expected_copy_number_sum, aspect='auto', cmap='RdBu_r', vmin=1, vmax=2.,
		#            interpolation='none', extent=[0, GlobalConstants.CHROM_WT1_TIMEPOINTS[-1], 
		#                                          0, len(gene_chr_counts)])
		# plt.colorbar()
		# plt.yticks([])
		# plt.ylabel("Genes sorted by replication")
		# plt.xlabel("Time, min")
		# plt.title("Est. Copy #", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)

		plt.subplot(1, 2, 1)
		plt.imshow((gene_chr_counts*scale+offset), aspect='auto', 
		          vmin=0, vmax=2, interpolation='none',
		          extent=[0, GlobalConstants.CHROM_WT1_TIMEPOINTS[-1], 0,
		                  len(gene_chr_counts)], cmap='RdBu_r')
		plt.colorbar()
		plt.yticks([])
		plt.xlabel("Time, min")
		plt.title("Raw occupancy", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)

		plt.subplot(1, 2, 2)
		normalized_corrected_counts = self.corrected_counts / \
		    self.corrected_counts.sum(axis=0).values.reshape((1, -1))
		plt.imshow(self.corrected_counts, aspect='auto', 
		          vmin=0, vmax=2, interpolation='none',
		          extent=[0, GlobalConstants.CHROM_WT1_TIMEPOINTS[-1], 0,
		                  len(gene_chr_counts)], cmap='RdBu_r')
		plt.colorbar()
		plt.yticks([])
		plt.xlabel("Time, min")
		plt.title("Corrected occupancy", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		plt.suptitle(f"Copy number correction, gene 10 kb occupancy, n={len(normalized_corrected_counts)}",
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)

	def compute_ptr(self):
		from src.peak_to_trough import compute_quantile_ptr_2d

		# For replicate 1, start with the third timepoint forwards to handle the recovery
		# G1 timepoints
		cols = self.gene_10k_counts.columns[3:]
		self.raw_ptrs = compute_quantile_ptr_2d(self.gene_10k_counts[cols])
		self.corrected_ptrs = compute_quantile_ptr_2d(self.corrected_counts[cols])


	def plot_ptr(self):
		from src.figure_configs import FiguresConfig

		plot_data = self.genes_repl_profile.copy()
		plot_data['raw_ptr'] = self.raw_ptrs
		plot_data['corrected_ptr'] = self.corrected_ptrs
		plot_data = plot_data.loc[self.genes.index] # Plot by genomic index
		self.ptr_df = plot_data

		plt.figure(figsize=(6, 5))
		plt.scatter(plot_data.raw_ptr, plot_data.corrected_ptr, s=3,
			c=plot_data.replication_time, cmap='inferno_r',
			vmin=5, vmax=16)
		plt.plot([0, 10], [0, 10], lw=1, ls='dotted', zorder=0, color='black')
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Replication time", rotation=270, va='bottom')

		plt.xlim(0.95, 2)
		plt.ylim(0.95, 2)
		plt.title("PTR correction, gene 10 kb windows", 
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE, pad=9)
		plt.xlabel("Uncorrected PTR")
		plt.ylabel("Corrected PTR")


	def plot_copy_correction_curves(self):

		replication_indices = self.genes_repl_profile.replication_H_index.values.astype(int)

		plt.figure(figsize=(13, 3))
		plt.subplot(1, 3, 1)
		plt.imshow(self.H_combined_copy_num, vmax=0.05, aspect='auto', interpolation='none')
		plt.colorbar()
		plt.title("H w/ expected copy number")
		
		num_curves = 100
		indices = np.linspace(0, n-1, num_curves).astype(int)
		colors = [plt.get_cmap('RdBu_r')(float(i)/n) for i in indices]

		n = len(replication_indices)
		plt.subplot(1, 3, 2)

		plt.plot(self.H_expected_copy_number_sum.T[:, 0:n:100], c='red', alpha=0.5)
		plt.title("Expected copy number")

		plt.subplot(1, 3, 3)
		plt.plot(self.H_normalized_copy_per_gene[:, 0:n:100], c='red', alpha=0.5)
		plt.title("Copy number per gene, normalized")

		
def create_copy_number_H(H, replication_idx):
	"""Create a copy number matrix from H, converting indices from the replication index
	onward to two copies.
	
	The resulting matrix is a modification of the original proportion matrix that represents
	the overall expected copy number per timepoint when the columns are collapsed
	"""
	c1_indices = np.concatenate([np.arange(replication_idx), np.array([H.shape[1]-1])])
	c2_indices = np.arange(replication_idx, H.shape[1]-1)

	# Combine the two for the expected copy number for the gene
	H_expected_copy_num = H.copy()
	H_expected_copy_num[:, c2_indices] = H[:, c2_indices]*2
	return H_expected_copy_num


def correct_replication_indices(H, replication_indices):
	"""Create a combined H matrix that includes each of the copy number 
	corrected H matrices.
	
	Then create a normalized copy number matrix per gene. A matrix that represents
	the copy correction including the normalizing effect of varying replication times
	per genome segment.
	"""
	num_genes = len(replication_indices)
	H_expected_copy_per_gene = np.zeros((num_genes, *H.shape))

	for i in range(num_genes):
		 H_expected_copy_per_gene[i] = create_copy_number_H(H, replication_indices[i])

	H_combined_copy_num = np.sum(H_expected_copy_per_gene, axis=0) / num_genes
	overall_sum = H_combined_copy_num.sum(axis=1)
	H_normalized_copy_per_gene = H_expected_copy_per_gene.sum(axis=2).T / \
		overall_sum.reshape((-1, 1))
	return H_combined_copy_num, H_expected_copy_per_gene, H_normalized_copy_per_gene
