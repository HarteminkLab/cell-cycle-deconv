
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.figure_configs import FiguresConfig
from src.global_config import GlobalConstants


class CopyCorrectionAnalysis:
	"""In this class, copy correction is revisited. The idea is to simplify the copy
	number correction process and clarify step-by-step how the process is performed:

	Procedure:
	1. Retrieve the 10 kb occupancy counts for all chromosomes
	2. Normalize these occupancy counts to match up the first cell cycle min and max 
		occupancy values. 
		Assupmption: All segments of the genome will reach these same minimum and maximal values
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
		config1, config2 = load_configs_by_config_type('shared', with_copy_correction=False)
		self.H1, Hpos = config1.calcH_function(config1.intervals_wt1, config1.WT1_TIMEPOINTS)
		self.H2, Hpos = config2.calcH_function(config2.intervals_wt1, config2.WT1_TIMEPOINTS)

		self.config1 = config1
		self.config2 = config2

		from src.geneset import get_deconvolved_geneset
		self.genes = get_deconvolved_geneset()


	def set_replicate(self, replicate, shuffle=False):

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

		self.chrom_replication_profile = pd.read_csv(
			'output/replication_profiles/chrom_replication_timing_shared.csv').set_index(['chr', 'start'])

		# Shuffle the replication profile for a proof of concept, that the correction is non-trivial
		if shuffle:
			print("Shuffling the replication index and times.")
			np.random.seed(123)
			shuffled_index = self.chrom_replication_profile.index.values.copy()
			np.random.shuffle(shuffled_index)
			self.chrom_replication_profile.replication_index = \
				self.chrom_replication_profile.loc[shuffled_index].replication_index.values

		self.chrom_bin_curves = chr_bin_curves

		# Add replication timing
		idx_tp_mapping = self.config.index_tp_mapping_df()
		repl_tp = [idx_tp_mapping.loc[repl_index].timepoint for repl_index in \
			self.chrom_replication_profile.replication_index.values]
		repl_profile = self.chrom_replication_profile
		repl_profile['replication_time'] = repl_tp
		self.chrom_replication_profile = repl_profile


	def scale_occupancy_curves(self):

		n = len(self.chrom_bin_curves)
		mixture_curves = self.chrom_bin_curves.copy()

		# The 10k occupancy scaled to match the expected H curves,
		# Early and late replicating genes have different max values
		data_scaled_10k = self.chrom_bin_curves.copy() 
		for i in range(n):
			mixture_curve, occ_curve_scaled = self.scale_occupancy_curves_index(i)
			mixture_curves.iloc[i] = mixture_curve
			data_scaled_10k.iloc[i] = occ_curve_scaled

		self.data_scaled_10k = data_scaled_10k
		self.mixture_curves = mixture_curves


	def scale_occupancy_curves_index(self, idx):
		"""Scale the occupancy curves to match the mixture curves, this will
		ensure the estimated copy number curve and observed 10k window are
		in the same scale range. Early vs late replicationg windows
		have slightly different max values.
		"""

		repl_idx = self.chrom_replication_profile.iloc[idx].replication_index.astype(int)

		repl_curve = np.ones(self.H.shape[1])
		repl_curve[repl_idx:-1] = 2

		occ_curve = self.chrom_bin_curves.iloc[idx]
		mixture_curve = self.H @ repl_curve
		value_range = mixture_curve.max()-mixture_curve.min()

		non_repl = np.ones_like(repl_curve)
		non_repl_curve = self.H @ non_repl

		# Scaled occupancy curves will be values from 1-(max of the copy mixture sum)
		occ_curve_scaled = occ_curve*value_range+mixture_curve.min()
		
		return mixture_curve, occ_curve_scaled


	def perform_correction(self):

		def normalize_cols(df):
			df = df / df.sum(axis=0).values.reshape((1 ,-1))
			return df

		mixture_curves = self.mixture_curves
		data_scaled_10k = self.data_scaled_10k

		self.normalized_mixture_curves = normalize_cols(mixture_curves) * len(mixture_curves)
		self.normalized_data_10k = normalize_cols(data_scaled_10k) * len(mixture_curves)


		self.norm_corrected = (self.normalized_data_10k-1) * \
			(self.normalized_mixture_curves-1)+1

		self.norm_norm_corrected = normalize_cols(self.norm_corrected)*len(mixture_curves)


	def plot_normalization_example_curves(self):

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
			line_early = Line2D([0], [0], lw=2, label=f'Early, chr{early_idx[0]}-{early_idx[1]}', color=color_early)
			line_late = Line2D([0], [0], lw=2, label=f'Late, chr{late_idx[0]}-{late_idx[1]}', color=color_late)
			line_mix = Line2D([0], [0], lw=1, ls=mixture_ls, 
				label='Est. Mixture', color='black')
			line_raw = Line2D([0], [0], lw=1, ls=raw_ls, label='Raw', color='black')
			line_corrected = Line2D([0], [0], lw=1, ls=corrected_ls, 
				label='Corrected', color='black')

			# add manual symbols to auto legend
			if include_corrected:
				handles.extend([line_mix, line_raw, line_corrected, 
					line_early, line_late])
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

		sorted_idx = self.chrom_replication_profile.sort_values('replication_index').index
		early_idx = sorted_idx.values[25]
		late_idx = sorted_idx.values[-25]

		mixture_curves = self.mixture_curves.loc[sorted_idx]
		data_scaled_10k = self.data_scaled_10k.loc[sorted_idx]
		normalized_mixture_curves = self.normalized_mixture_curves.loc[sorted_idx]
		normalized_data_10k = self.normalized_data_10k.loc[sorted_idx]
		norm_corrected = self.norm_corrected.loc[sorted_idx]
		norm_norm_corrected = self.norm_norm_corrected.loc[sorted_idx]

		plt.subplot(1, 2, 1)
		plot_mix_data(mixture_curves.loc[early_idx], data_scaled_10k.loc[early_idx],
					  color=color_early)
		plot_mix_data(mixture_curves.loc[late_idx], data_scaled_10k.loc[late_idx],
					  color=color_late)
		plt.title("Unnormalized")
		plt.ylabel("Copy #")
		create_legend(False)

		plt.subplot(1, 2, 2)
		plot_mix_data(normalized_mixture_curves.loc[early_idx],
					  normalized_data_10k.loc[early_idx], 
					  norm_norm_corrected.loc[early_idx],
					  color=color_early)
		plot_mix_data(normalized_mixture_curves.loc[late_idx], 
					  normalized_data_10k.loc[late_idx],
					  norm_norm_corrected.loc[late_idx],
					  color=color_late)
		plt.title("Equal sample normalization")
		plt.ylabel("Normalized copy #")

		plt.ylabel("Normalized copy #")
		plt.title("Normalization+Correction", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		plt.suptitle(f"Replicate {self.replicate}", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.85)

		create_legend(True)


	def plot_heatmap_correction(self):
		plt.figure(figsize=(7, 6))
		plt.subplot(1, 2, 1)

		sorted_idx = self.chrom_replication_profile.sort_values('replication_index').index

		sorted_10K_occ = self.data_scaled_10k.loc[sorted_idx]
		norm_mix = self.normalized_mixture_curves.loc[sorted_idx]
		norm_raw = self.normalized_data_10k.loc[sorted_idx]
		norm_corrected = self.norm_corrected.loc[sorted_idx]

		extent = [0, self.tps[-1], 0, len(norm_mix)]

		plt.imshow(sorted_10K_occ, aspect='auto', cmap='RdBu_r', 
					interpolation='none', vmin=0, vmax=2., 
					extent=extent)
		plt.yticks([])
		#plt.colorbar()
		plt.title("10k occupancy", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)
		plt.ylabel("Windows sorted by replication timing")
		plt.xlabel("Time, min")

		plt.subplot(1, 2, 2)
		plt.imshow(norm_raw, aspect='auto', cmap='RdBu_r', 
					interpolation='none', vmin=0, vmax=2, 
					extent=extent)
		plt.yticks([])
		plt.xlabel("Time, min")
		plt.colorbar()
		plt.title("10k occupancy,\nnormalized", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)
		plt.suptitle(f"Replicate {self.replicate}", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.85)


	def compute_ptr_correction(self):
		from src.peak_to_trough import compute_quantile_ptr_2d

		raw_ptr = compute_quantile_ptr_2d(self.normalized_data_10k, return_indices=False)
		corrected_ptr = compute_quantile_ptr_2d(self.norm_corrected, return_indices=False)

		ptr_df = pd.DataFrame({
			'raw': raw_ptr, 
			'corrected': corrected_ptr, 
			'replication_time': self.chrom_replication_profile.replication_time
		})

		# Sort by genome
		self.ptr_df = ptr_df

		vals_30 = self.normalized_data_10k[30]
		vals_0 = self.normalized_data_10k[0]
		corrected_30 = self.norm_corrected[30]
		corrected_0 = self.norm_corrected[0]

		def compute_ratio_gt_over_lt(vals_30, vals_0):
			ratio_vals = vals_0.copy()
			ratio_vals[vals_0.values > vals_30.values] = vals_0/vals_30
			ratio_vals[vals_0.values <= vals_30.values] = vals_30/vals_0
			return ratio_vals

		ratio_vals = compute_ratio_gt_over_lt(vals_30, vals_0)
		ratio_corrected = compute_ratio_gt_over_lt(corrected_30, corrected_0)

		ratios_30_0_df = self.ptr_df.copy()
		ratios_30_0_df['ratio_raw'] = ratio_vals
		ratios_30_0_df['ratio_corrected'] = ratio_corrected
		self.ratios_30_0_df = ratios_30_0_df


	def plot_ptr_scatter(self):
		ptr_df = self.ptr_df
		ratios_30_0_df = self.ratios_30_0_df

		plt.figure(figsize=(14, 4.75))

		plt.subplot(1, 2, 1)
		plt.scatter(ratios_30_0_df.ratio_raw, ratios_30_0_df.ratio_corrected, edgecolor='#aaa',
					facecolors='none', s=3)
		plt.scatter(ratios_30_0_df.ratio_raw, ratios_30_0_df.ratio_corrected, 
					c=ratios_30_0_df.replication_time, cmap='inferno_r', s=1, vmin=6, vmax=16)
		plt.xlim(0.995, 1.6)
		plt.ylim(0.995, 1.05)
		plt.plot([0, 2], [0, 2], lw=1, c='gray', zorder=0, ls=(1, (1, 1)))
		plt.colorbar()
		plt.title("Raw vs Corrected, Ratio of 30' & 0'")
		plt.xlabel("Ratio max[30', 0'] / min[30', 0'], raw")
		plt.ylabel("Ratio max[30', 0'] / min[30', 0'], corrected")


		plt.subplot(1, 2, 2)
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


	def compute_scaling_term(self, save=True):
		"""The scaling will be based on the unnormalized bins (not normalized by the overall copy
		number curve). However, we will be normalizing the bins of the chromatin by length
		distribution, which means we want all the samples to be of equal size. So
		scale the bins such that each timepoint sums to the same value. Then
		compute the scaling term for the gene and origin deconvolutions
		"""

		if self.replicate == 1:
			unnormalized_bins = self.mnase_occupancies_1.unnormalized_chr_bin_curves
		else:
			unnormalized_bins = self.mnase_occupancies_2.unnormalized_chr_bin_curves

		equal_scaled_bins = unnormalized_bins / unnormalized_bins.sum(axis=0).values.reshape((1, -1))\
			* len(unnormalized_bins)

		scaling_term = equal_scaled_bins / self.norm_norm_corrected
		self.scaling_term = scaling_term

		if save:
			save_path = f'data/copy_correction/chromatin/copy_correction_replicate{self.replicate}.csv'
			scaling_term.reset_index().to_csv(save_path, index=False)
			print("Saved to: ", save_path)


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
