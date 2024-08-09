
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

		self.config1 = config1
		self.config2 = config2

		from src.geneset import get_deconvolved_geneset
		self.genes = get_deconvolved_geneset()

	def compute_gene_10k_counts(self, replicate):

		self.replicate = replicate

		if replicate == 1:
			self.H = self.H1
			self.config = self.config1
			self.tps = GlobalConstants.CHROM_WT1_TIMEPOINTS
		else:
			self.H = self.H2
			self.config = self.config2
			self.tps = GlobalConstants.CHROM_WT2_TIMEPOINTS

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



	def scale_occupancy_curves(self):

		n = len(self.genes_repl_profile)
		mixture_curves = self.gene_10k_counts.copy()

		# The 10k occupancy scaled to match the expected H curves,
		# Early and late replicating genes have different max values
		data_scaled_10k = self.gene_10k_counts.copy() 
		for i in range(n):
			mixture_curve, occ_curve_scaled = self.scale_occupancy_curves_index(i)
			mixture_curves.iloc[i] = mixture_curve
			data_scaled_10k.iloc[i] = occ_curve_scaled

		self.data_scaled_10k = data_scaled_10k
		self.mixture_curves = mixture_curves


	def scale_occupancy_curves_index(self, gene_idx):
		"""Scale the occupancy curves to match the mixture curves, this will
		ensure the estimated copy number curve and observed 10k window are
		in the same scale range. Early vs late replicationg windows
		have slightly different max values.
		"""

		repl_idx = self.genes_repl_profile.iloc[gene_idx]\
			.replication_H_index.astype(int)

		repl_curve = np.ones(self.H.shape[1])
		repl_curve[repl_idx:-1] = 2

		occ_curve = self.gene_10k_counts.iloc[gene_idx]
		mixture_curve = self.H @ repl_curve
		value_range = mixture_curve.max()-mixture_curve.min()

		non_repl = np.ones_like(repl_curve)
		non_repl_curve = self.H @ non_repl

		# Scaled occupancy curves will be values from 1-(max of the copy mixture sum)
		occ_curve_scaled = occ_curve*value_range+1.
		
		return mixture_curve, occ_curve_scaled


	def perform_correction(self):

		def normalize_cols(df):
			df = df / df.sum(axis=0).values.reshape((1 ,-1))
			return df

		mixture_curves = self.mixture_curves
		data_scaled_10k = self.data_scaled_10k

		self.normalized_mixture_curves = normalize_cols(mixture_curves) * len(mixture_curves)
		self.normalized_data_10k = normalize_cols(data_scaled_10k) * len(mixture_curves)
		self.norm_corrected = (self.normalized_data_10k-1) * (self.normalized_mixture_curves-1)+1


	def plot_normalization_example_curves(self):

		early_idx = 25
		late_idx = -25

		color_early = plt.get_cmap('inferno_r')(0.2)
		color_late = plt.get_cmap('inferno_r')(0.8)

		mixture_ls=(0, (5, 2))
		raw_ls='solid'
		corrected_ls=(0, (1, 1))

		def create_legend(include_corrected=False):
			from matplotlib.lines import Line2D
			import matplotlib.patches as mpatches
			handles, labels = plt.gca().get_legend_handles_labels()

			# create manual symbols for legend
			line_early = Line2D([0], [0], lw=2, label='Early', color=color_early)
			line_late = Line2D([0], [0], lw=2, label='Late', color=color_late)
			line_mix = Line2D([0], [0], lw=1, ls=mixture_ls, label='Est. Mixture', color='black')
			line_raw = Line2D([0], [0], lw=1, ls=raw_ls, label='Raw', color='black')
			line_corrected = Line2D([0], [0], lw=1, ls=corrected_ls, label='Corrected', color='black')

			# add manual symbols to auto legend
			if include_corrected:
				handles.extend([line_mix, line_raw, line_corrected, line_early, line_late])
			else:
				handles.extend([line_mix, line_raw, line_early, line_late])
			
			plt.legend(handles=handles, ncol=2, loc='lower left')

		def plot_mix_data(mix, dat, corrected=None, color=None):
			plt.plot(self.tps, mix, label="_Copy number mixture", color=color,
					lw=1, ls=mixture_ls)
			plt.plot(self.tps, dat, label="_10k Occupancy", color=color,
					ls=raw_ls, lw=1)
			if corrected is not None:
				plt.plot(self.tps, corrected,
					 label="_Corrected occupancy", color=color, ls=corrected_ls, lw=1)
			plt.xlabel("Time, min")
			plt.ylim(0.4, 1.7)
			plt.axhline(1, c='#ddd', ls='solid', zorder=0, lw=0.5)

		plt.figure(figsize=(10, 4))

		mixture_curves = self.mixture_curves
		data_scaled_10k = self.data_scaled_10k
		normalized_mixture_curves = self.normalized_mixture_curves
		normalized_data_10k = self.normalized_data_10k
		norm_corrected = self.norm_corrected
			
		plt.subplot(1, 2, 1)
		plot_mix_data(mixture_curves.iloc[early_idx], data_scaled_10k.iloc[early_idx],
					  color=color_early)
		plot_mix_data(mixture_curves.iloc[late_idx], data_scaled_10k.iloc[late_idx],
					  color=color_late)
		plt.title("Unnormalized")
		plt.ylabel("Copy #")
		create_legend(False)

		plt.subplot(1, 2, 2)
		plot_mix_data(normalized_mixture_curves.iloc[early_idx],
					  normalized_data_10k.iloc[early_idx], 
					  norm_corrected.iloc[early_idx],
					  color=color_early)
		plot_mix_data(normalized_mixture_curves.iloc[late_idx], 
					  normalized_data_10k.iloc[late_idx],
					  norm_corrected.iloc[late_idx],
					  color=color_late)
		plt.title("Equal sample normalization")
		plt.ylabel("Normalized copy #")

		plt.ylabel("Normalized copy #")
		plt.title("Correction")
		create_legend(True)


	def plot_heatmap_correction(self):
		plt.figure(figsize=(13, 6))
		plt.subplot(1, 3, 1)

		norm_mix = self.normalized_mixture_curves
		norm_raw = self.normalized_data_10k
		norm_corrected = self.norm_corrected

		plt.imshow(norm_mix, aspect='auto', cmap='RdBu_r', 
					interpolation='none', vmin=.5, vmax=1.5)
		plt.yticks([])
		plt.colorbar()
		plt.title("Est. copy change")

		plt.subplot(1, 3, 2)
		plt.imshow(norm_raw, aspect='auto', cmap='RdBu_r', 
					interpolation='none', vmin=.5, vmax=1.5)
		plt.yticks([])
		plt.colorbar()
		plt.title("Raw")

		plt.subplot(1, 3, 3)
		plt.imshow(norm_corrected, 
				   aspect='auto', cmap='RdBu_r', 
				   interpolation='none', vmin=0.5, vmax=1.5)
		plt.colorbar()
		plt.yticks([])
		plt.title("Corrected")


	def compute_ptr_correction(self):
		from src.peak_to_trough import compute_quantile_ptr_2d

		raw_ptr = compute_quantile_ptr_2d(self.normalized_data_10k)
		corrected_ptr = compute_quantile_ptr_2d((self.norm_corrected))

		ptr_df = pd.DataFrame({
			'raw': raw_ptr, 'corrected': corrected_ptr, 
			'replication_time': self.genes_repl_profile.replication_time
		})

		# Sort by genome
		self.ptr_df = ptr_df.loc[self.genes.index]


	def plot_ptr_scatter(self):
		ptr_df = self.ptr_df

		plt.figure(figsize=(6, 4.75))
		plt.scatter(ptr_df.raw, ptr_df.corrected, s=3, 
					facecolors='none', edgecolor='#aaa', zorder=0)
					
		plt.scatter(ptr_df.raw, ptr_df.corrected, s=1, 
		c=ptr_df.replication_time, cmap='inferno_r', 
					vmin=6, vmax=16, zorder=1)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Replication time, min", rotation=270, va='bottom')
		plt.plot([0, 2], [0, 2], zorder=0, c='black', ls='dotted', lw=1)
		plt.xlim(0.995, 1.2)
		plt.ylim(0.995, 1.02)
		plt.xlabel("PTR, raw")
		plt.ylabel("PTR, corrected")
		plt.title("Raw vs Corrected PTR")
