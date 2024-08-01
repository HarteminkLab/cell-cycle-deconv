
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure_configs import FiguresConfig
from src.figure_configs import save_figure_for_paper
from src.plot_helpers import adjust_lightness_saturation


class Figure3CopyCorrection():
	"""Load and plot figures for the third result figure"""

	def __init__(self, shuffle=False):

		# Chromatin
		self.chrom_ptrs_rep1 = pd.read_csv('output/copy_correction/chromatin/ptrs_rep1_shared.csv')\
			.set_index('orf_name')
		self.chrom_ptrs_rep2 = pd.read_csv('output/copy_correction/chromatin/ptrs_rep2_shared.csv')\
			.set_index('orf_name')
		self.mean_chrom_ptrs = (self.chrom_ptrs_rep1 + self.chrom_ptrs_rep2) / 2.
		self.mean_chrom_ptrs['log_ratio'] = \
			np.log2(self.mean_chrom_ptrs['normalized_corrected_ptr'] / self.mean_chrom_ptrs['normalized_raw_ptr']+1)
		self.delta_chrom = 0.01

		# Selected genes based on increase/decrease in PTR
		self.selected_chrom_genes = ['RPA34', # Increase
									 'DUN1', # Decrease, Late
									 'APA1', # Decrease, Early
									 'SSK22' # Control gene
		]

		# Gene expression

		self.ge_ptrs_rep1 = pd.read_csv('output/copy_correction/gene_expression/expression_ptr_comparison_rep1_shared.csv')\
			.set_index('orf_name')
		self.ge_ptrs_rep2 = pd.read_csv('output/copy_correction/gene_expression/expression_ptr_comparison_rep2_shared.csv')\
			.set_index('orf_name')
		self.mean_ge_ptrs = (self.ge_ptrs_rep1 + self.ge_ptrs_rep2) / 2.
		self.mean_ge_ptrs['log_ratio'] = \
			np.log2(self.mean_ge_ptrs['corrected_ptr'] / self.mean_ge_ptrs['raw_ptr']+1)
		self.delta_ge = 0.007

		# Selected genes based on increase/decrease in PTR
		# Plus control genes CLB2 and SSK22
		self.selected_tx_genes = ['CHL4', 'NCA3', 'CSM1', # Increase
									 'HTA2', 'HTB2', 'DSE3', 'STU2', # Decrease
									 'CLB2' # Control genes
		]

		def add_repl_timing_group_id(ptrs_data_df):
			"""Add group ids for the replication timing, will be useful for box plots"""

			from src.helpers import get_quantile_values

			segments, qvals, lens = get_quantile_values(ptrs_data_df.replication_time, 
				q=[0.25, 0.5, 0.75])

			early_genes, early_mid_genes, mid_late_genes, late_genes = segments

			ptrs_data_df.loc[early_genes.index, 'group_id'] = 0
			ptrs_data_df.loc[early_mid_genes.index, 'group_id'] = 1
			ptrs_data_df.loc[mid_late_genes.index, 'group_id'] = 2
			ptrs_data_df.loc[late_genes.index, 'group_id'] = 3

			ptrs_data_df.loc[early_genes.index, 'group_name'] = 'Early'
			ptrs_data_df.loc[early_mid_genes.index, 'group_name'] = 'Early-Mid'
			ptrs_data_df.loc[mid_late_genes.index, 'group_name'] = 'Mid-Late'
			ptrs_data_df.loc[late_genes.index, 'group_name'] = 'Late'

		# Add group information for box plotting
		add_repl_timing_group_id(self.mean_ge_ptrs)
		add_repl_timing_group_id(self.mean_chrom_ptrs)


	def plot_S_diagram(self):
		"""Create a diagram that depicts the distribution of origin of replications to convey
		the variability in genomic replication timing through S phase"""

		from src.config import load_configs_by_config_type
		from src.origins import load_origins_w_replication

		# Copy number correction procedure....
		config1, config2 = load_configs_by_config_type('shared', 1)

		chrom_replication_profile = pd.read_csv(
			'data/replication_timing/yl_2019/chrom_replication_timing_shared.csv')
		chrom_replication_profile = chrom_replication_profile.set_index(['chr', 'start'])

		t = config1.get_timepoints_for_branch('t')
		t_indices = config1.get_Hpositions_for_branch('t')
		s_indices = config1.get_Hpositions_for_phase('S')
		cg1_indices = config1.get_Hpositions_for_phase('CG1')
		t_tps = (config1.get_timepoints_for_branch('t')+config2.get_timepoints_for_branch('t'))/2
		s_tps = t_tps[len(cg1_indices):(len(cg1_indices)+len(s_indices))]

		origins = load_origins_w_replication(full=True)
		origins = origins[origins.footprint_class == 'g1_and_g2_footprint']
		g1 = (config1.get_g1_lens('CG1') + config2.get_g1_lens('CG1'))/2

		t_index_tp_mapping = {}
		for i in range(len(t_indices)):
			t_index_tp_mapping[t_indices[i]] = t_tps[i]

		from scipy.stats.distributions import norm

		# replication_indices = chrom_replication_profile.values.flatten()
		replication_tps = []

		eff_key = 'derived_origin_efficiency_from_mcguffee_et_al_2013'

		for origin_name, origin in origins.iterrows():
			# add one to plot the first index in which copy number is 2
			rep_tp = t_index_tp_mapping[origin.replication_index+1]+g1
			replication_tps.append(rep_tp)

		replication_tps = np.array(replication_tps) +\
			norm.rvs(loc=0, scale=0.25, size=len(replication_tps))

		from src.chromatin_model import draw_phase_label_annotations
		from src.config import load_configs_by_config_type
		config1, config2 = load_configs_by_config_type('shared')

		plt.figure(figsize=(4, 2))
		ax = plt.gca()
		draw_phase_label_annotations(ax, config1, flip=True, offset=True, annotations_x=-0.5)

		plt.yticks([])

		ys = np.repeat(1.0, len(replication_tps))

		alphas = 1.#origins[eff_key].values
		#alphas[alphas > 1.] = 1
		#alphas[alphas < 0] = 0
		plt.axhline(1, c='black', lw=1.5)
		for i in range(len(ys)):
			tp = replication_tps[i]
			y = ys[i]
			alpha = alphas[i]#*1#0.5
			plt.plot([tp, tp], [y-0.3, y+0.3], alpha=alpha, lw=0.5, 
				color=plt.get_cmap('Reds')(0.6))
		plt.xlim(16, 45)
		plt.ylim(-1, 10)

		from src.plot_helpers import hide_spines
		xs = np.arange(0, 100, 0.25)
		ys = norm.pdf(xs, loc=30, scale=3)*50.
		plt.fill_between(xs, 1, ys+1, color='#dddddd', zorder=0)




		hide_spines(plt.gca())


	def plot_replication_profile_example(self):

		from src.config import load_configs_by_config_type
		from src.origins import load_origins_w_replication

		origins = load_origins_w_replication(full=True)
		origins = origins[origins.footprint_class == 'g1_and_g2_footprint']

		# Depiction of the chromosome 10 replication timing profile computed from the MNase-seq
		chrom_replication_profile = pd.read_csv(
			'data/replication_timing/yl_2019/chrom_replication_timing_shared.csv')
		chrom_replication_profile = chrom_replication_profile.set_index(['chr', 'start'])

		# Copy number correction procedure....
		config1, config2 = load_configs_by_config_type('shared', 1)

		chrom = 10

		from src.stepwise_replication_solver import replication_timing_from_index

		repl_idx = chrom_replication_profile.loc[chrom].replication_index
		repl_timing = replication_timing_from_index(repl_idx, config1, config2)

		t = config1.get_timepoints_for_branch('t')
		t_indices = config1.get_Hpositions_for_branch('t')
		s_indices = config1.get_Hpositions_for_phase('S')
		cg1_indices = config1.get_Hpositions_for_phase('CG1')
		t_tps = (config1.get_timepoints_for_branch('t')+config2.get_timepoints_for_branch('t'))/2
		s_tps = t_tps[len(cg1_indices):(len(cg1_indices)+len(s_indices))]

		t_index_tp_mapping = {}
		for i in range(len(t_indices)):
			t_index_tp_mapping[t_indices[i]] = t_tps[i]

		g1 = (config1.get_g1_lens('CG1') + config2.get_g1_lens('CG1'))/2

		remade_f = np.zeros((len(repl_idx), len(t_indices)))+1

		pd.DataFrame(remade_f, index=repl_idx.index)
		time_indices = np.arange(remade_f.shape[1])
		for i in range(remade_f.shape[0]):
			remade_f[i, time_indices > repl_idx.iloc[i]] = 2
			
		plt.figure(figsize=(9, 4.5))

		# Choose efficient origins to plot
		# 455 origins from Belsky data with cross referenced efficiency values from McGuffee (filter out
		# any -1 efficiency values)
		origins_chr = origins[origins.chr == chrom].copy()

		from src.global_config import GlobalConstants

		replication_img_in_S = remade_f.T[s_indices]

		extents = [0, repl_idx.index[-1], 
			s_tps[0]+g1, s_tps[-1]+g1]
		plt.subplot(2, 1, 1)
		plt.imshow(replication_img_in_S, aspect='auto', cmap='Blues',
			extent=extents, vmin=1, vmax=2, origin='lower')
		plt.text(extents[1]*1/20., extents[2]+3.5, '1 Copy')
		plt.text(extents[1]*1/20., extents[3]+2, '2 Copies', color='white')
		plt.ylabel("Replication time, min", fontsize=11)
		plt.xlabel("Genomic position, nt", fontsize=11, labelpad=7)

		# Add padding to the replication profile for display purposes
		updated_extents = extents[0], extents[1], extents[2], extents[3]+5
		plt.ylim(updated_extents[3], updated_extents[2])
		plt.imshow(np.array([[2]]), aspect='auto', cmap='Blues',
			extent=updated_extents, vmin=1, vmax=2, origin='lower', zorder=-1)

		from src.sgd import get_chromosome_length
		from src.mnase_replication_timing_analysis import get_bin_for_position

		eff_key = 'derived_origin_efficiency_from_mcguffee_et_al_2013'
		origins_chr['alpha'] = 1.0 # origins_chr[eff_key] + 0.2
		# origins_chr.loc[origins_chr['alpha'] < 0.2, 'alpha'] = 0.2 # Span values from 0.2-1.0,
		# original values are from 0-0.75 (with -1's as not found)

		for origin_name, origin in origins_chr.iterrows():
			# add one to plot the first index in which copy number is 2
			rep_tp = t_index_tp_mapping[origin.replication_index+1]+g1
			plt.scatter(origin.pos-GlobalConstants.REPL_DECONV_BIN_WIDTH/2., rep_tp,
				s=25, facecolors='white', lw=1.5, color='red', alpha=origin.alpha, zorder=100)

		xlims = extents[0], extents[1]
		xticks = np.arange(0, xlims[1], 100000)
		xticklabels = [f"{x}" for x in xticks]
		plt.xticks(xticks, xticklabels)
		xticks = np.arange(0, xlims[1], 50000)
		plt.xticks(xticks, minor=True)
		plt.xlim(0, xlims[1])

		plt.title(f"Deconvolved replication profile, chr{chrom}", 
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE, pad=9)

	def plot_raw_replication_chr10(self):
		from src.stepwise_replication_solver import StepReplicationChromatinDeconvolveSolver
		from src.mnase_replication_timing_analysis import MNaseOriginAnalysis

		mnase_analysis_rep1 = MNaseOriginAnalysis(replicate=1)
		mnase_analysis_rep1.compute_bin_curves(chroms=[10])

		mnase_analysis_rep2 = MNaseOriginAnalysis(replicate=2)
		mnase_analysis_rep2.compute_bin_curves(chroms=[10])

		mnase_analysis_rep1.normalize_and_compute_raw_replication_timing()
		mnase_analysis_rep2.normalize_and_compute_raw_replication_timing()

		solver = StepReplicationChromatinDeconvolveSolver(mnase_analysis_rep1,
														  mnase_analysis_rep2,
														 config_type='shared')
		solver.set_chrom(10)
		solver.plot_raw_data()

	def plot_chrom_ptr_correction(self, selected_genes=None, shuffled=False):
		
		if shuffled:

			# Chromatin
			chrom_ptrs_rep1 = pd.read_csv('output/copy_correction_shuffled/chromatin/ptrs_rep1_shared.csv')\
				.set_index('orf_name')
			chrom_ptrs_rep2 = pd.read_csv('output/copy_correction_shuffled/chromatin/ptrs_rep2_shared.csv')\
				.set_index('orf_name')
			mean_chrom_ptrs = (chrom_ptrs_rep1 + chrom_ptrs_rep2) / 2.

			plot_data = mean_chrom_ptrs 
		else:
			plot_data = self.mean_chrom_ptrs

		plt.figure(figsize=FiguresConfig.FIGSIZE_SQUARE_WIDE)
		plt.scatter(plot_data.normalized_raw_ptr, 
					plot_data.normalized_corrected_ptr,
					facecolors='none', lw=2, edgecolor='#ddd', s=2, zorder=1)
		cmap = ptr_cmap()
		plt.scatter(plot_data.normalized_raw_ptr, 
					plot_data.normalized_corrected_ptr,
					c=plot_data.replication_time,
					vmax=15,
					s=1, cmap=cmap, zorder=2)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Replication time, min", rotation=270, va='bottom')

		n = len(plot_data)
		if shuffled:
			title = f"Mean Chromatin PTR (shuffled replication timing),\nRaw vs Copy # Corrected n={n}"
		else:
			title = f"Mean Chromatin PTR,\nRaw vs Copy # Corrected n={n}"

		plt.title(title, fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.plot([1, 2], [1, 2], c='black', lw=0.5, ls='dotted', zorder=10)

		xlim, ylim = (1, 1.5), (1, 1.5)
		plt.xlim(*xlim)
		plt.ylim(*ylim)

		if selected_genes is None:
			selected_genes = self.selected_chrom_genes

		plot_selected_genes(plot_data, selected_genes, 'normalized_raw_ptr', 'normalized_corrected_ptr',
			xlim, ylim)
		
		plt.xlabel("Raw chromatin occupancy PTR")
		plt.ylabel("Copy-number-corrected chromatin occupancy PTR")

	def plot_violin_ptr_11(self, gene_groups=None):
		"""Compute the distance of each ptr adjustment from the 1:1 line. Then create a violin plot
		of the distribution of these values for each replication timing window."""


		def distance_to_line(point, a=1, b=-1, c=0):
			"""Compute the distance of a point to a line as defined by: ax + by = c.
			Positive/negative distance correspond to point's relationship to the line. In
			the 1:1 case, positive values indicate x > y and thus below the line.
			"""
			
			x, y = point
			numerator = a*x + b*y + c
			denominator = a**2 + b**2
			
			distance = numerator / np.sqrt(denominator)
			
			return distance

		# Compute a difference from the 1:1 curve for each gene to designate
		# how the copy correction affected the gene's cyclicity as a single value.
		ptrs = self.mean_chrom_ptrs

		ptr_distances_11 = ptrs[['normalized_raw_ptr', 'normalized_corrected_ptr',
								 'group_id', 'group_name', 'log_ratio']].copy()
		ptr_values = ptrs[['normalized_raw_ptr', 'normalized_corrected_ptr']].values
		distances = np.apply_along_axis(distance_to_line, axis=1, arr=ptr_values)
		# Negative values will associate with, a decrease in PTR following correction
		ptr_distances_11['distance_11'] = -distances 
		from src.violinplot import ViolinPlotPlotter

		dat = ptr_distances_11
		violin_plotter = ViolinPlotPlotter()
		violin_plotter.set_data([dat], data_key='distance_11', group_key='group_id',
						   group_name_key='group_name', category_names=[''])
		ax = plt.gca()
		title = "Copy correction, Distance from 1:1"

		from src.plot_helpers import adjust_lightness_saturation

		violin_plotter.legend = False
		violin_plotter.group_colors = [
			adjust_lightness_saturation(plt.get_cmap('RdBu')(0.), 1.7, 1.0),
			plt.get_cmap('RdBu')(0.25),
			plt.get_cmap('RdBu')(0.75),
			adjust_lightness_saturation(plt.get_cmap('RdBu')(1.), 1.7, 1.0),
		]
		violin_plotter.plot_box_plot(ax=ax, title='')
		ax.set_ylim(-0.05, 0.05)
		ax.axhline(0, c='black', lw=0.75, zorder=-1, ls='dotted')
		ax.set_ylabel("PTR adjustment change, distance from 1:1")
		ax.set_title(title, fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		self.mean_chrom_ptrs_w_distances_11 = ptr_distances_11
		self.violin_plotter = violin_plotter


	def plot_ge_ptr_correction(self, selected_genes=None):
		
		plot_data = self.mean_ge_ptrs
		plt.figure(figsize=FiguresConfig.FIGSIZE_SQUARE_WIDE)
		plt.scatter(plot_data.raw_ptr, 
					plot_data.corrected_ptr,
					facecolors='none', lw=2, edgecolor='#ddd', s=2, zorder=1)
		cmap = ptr_cmap()
		plt.scatter(plot_data.raw_ptr, 
					plot_data.corrected_ptr,
					c=plot_data.replication_time,
					vmax=15,
					s=1, cmap=cmap, zorder=2)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Replication time, min", rotation=270, va='bottom')

		n = len(plot_data)
		plt.title(f"Gene expression PTR,\nRaw vs Copy # Corrected n={n}", 
			fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.plot([0, 2], [0, 2], c='black', lw=0.5, ls='dotted', zorder=10)

		xlim = 0.99, 1.2
		ylim = 0.99, 1.2

		if selected_genes is None:
			selected_genes = self.selected_tx_genes

		plot_selected_genes(plot_data, selected_genes, 'raw_ptr', 'corrected_ptr',
			xlim, ylim)

		plt.xticks(np.arange(1, 1.25, 0.05))
		plt.yticks(np.arange(1, 1.25, 0.05))

		plt.xlim(*xlim)
		plt.ylim(*ylim)
		
		plt.xlabel("Raw expression PTR")
		plt.ylabel("Copy-number-corrected expression PTR")


	def plot_chrom_ptr_rep_quantiles(self):

		chrom_ptrs = self.mean_chrom_ptrs.copy()
		n = len(chrom_ptrs)

		from src.boxplot import BoxPlotPlotter

		box_plotter = BoxPlotPlotter()
		box_plotter.set_data([chrom_ptrs.sort_values('replication_time'),
							 ], data_key='log_ratio', 
							 group_key='group_id',
							 group_name_key='group_name',
							category_names=[
								''])
		box_plotter.legend = False
		box_plotter.group_colors = rep_quantile_colors()
		box_plotter.width = 0.25

		fig = plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT)
		box_plotter.plot_box_plot(ax=plt.gca(), title='')
		plt.ylabel("Log2-ratio change in PTR")
		plt.title(f"Chromatin copy correction\nchange in PTR, n={n}", 
				  fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.axhline(1, c='black', lw=0.5, ls='solid', zorder=0)

		plt.ylim(-0.07, 0.07)
		plt.ylim(0.95, 1.05)

		plt.axhline(1-self.delta_chrom, c='#ddd', lw=0.7, ls='solid')
		plt.axhline(1+self.delta_chrom, c='#ddd', lw=0.7, ls='solid')

	def plot_chrom_ptr_counts(self):
		from src.Figure3_Copy_Correction import rep_quantile_colors

		plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT)

		ptrs = self.mean_chrom_ptrs

		group_names = ['Early', 'Early-Mid', 'Mid-Late', 'Late']

		for i in range(len(group_names)):
			group_name = group_names[i]
			current_ptrs = ptrs[ptrs.group_name == group_name]
			color = rep_quantile_colors()[i]
			
			from src.plot_helpers import adjust_lightness_saturation
			neg_color = adjust_lightness_saturation(color, 0.6, 0.75)
			
			plt.bar(i, -len(current_ptrs[current_ptrs.log_ratio < 1-self.delta_chrom]), color=neg_color, 
					width=0.3)
			plt.bar(i, len(current_ptrs[current_ptrs.log_ratio > 1+self.delta_chrom]), color=color, 
					width=0.3)

		plt.xticks(np.arange(len(group_names)), group_names)
		yticks = np.arange(-1000, 800, 100)
		ytick_labels = [str(np.abs(y)) for y in yticks]
		plt.yticks(yticks, ytick_labels)
		plt.ylim(-420, 100)
		plt.xlim(-0.5, 3.5)
		plt.axhline(0, c='black', lw=1, zorder=0)
		plt.ylabel("# of genes w/\nincreased or decreased PTR")
		plt.title(f"Genes with increased/decreased \nChromatin PTR "
				  f"(±{self.delta_chrom}) after correction",
				 pad=10, fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)


	def plot_ge_ptr_rep_quantiles(self):
		from src.boxplot import BoxPlotPlotter

		chrom_ptrs = self.mean_chrom_ptrs
		n = len(chrom_ptrs)

		box_plotter = BoxPlotPlotter()
		box_plotter.set_data([chrom_ptrs.sort_values('replication_time'),
							 ], data_key='log_ratio', 
							 group_key='group_id',
							 group_name_key='group_name',
							category_names=[
								''])
		box_plotter.legend = False
		box_plotter.group_colors = rep_quantile_colors()
		box_plotter.width = 0.25


		fig = plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT)
		box_plotter.plot_box_plot(ax=plt.gca(), title='')
		plt.ylabel("Log2-ratio change in PTR")
		plt.title(f"Gene expression copy correction\nchange in PTR, n={n}", 
				  fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.axhline(1, c='black', lw=0.5, ls='solid', zorder=0)

		plt.ylim(-0.07, 0.07)
		plt.ylim(0.96, 1.05)
		plt.axhline(1-self.delta_ge, c='#ddd', lw=0.7, ls='solid')
		plt.axhline(1+self.delta_ge, c='#ddd', lw=0.7, ls='solid')


	def plot_gene_expression_ptr_counts(self):
		from src.Figure3_Copy_Correction import rep_quantile_colors

		plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT)

		ptrs = self.mean_ge_ptrs

		group_names = ['Early', 'Early-Mid', 'Mid-Late', 'Late']

		for i in range(len(group_names)):
			group_name = group_names[i]
			current_ptrs = ptrs[ptrs.group_name == group_name]
			color = rep_quantile_colors()[i]
			
			from src.plot_helpers import adjust_lightness_saturation
			neg_color = adjust_lightness_saturation(color, 0.6, 0.75)
			
			plt.bar(i, -len(current_ptrs[current_ptrs.log_ratio < 1-self.delta_ge]), color=neg_color, 
					width=0.3)
			plt.bar(i, len(current_ptrs[current_ptrs.log_ratio > 1+self.delta_ge]), color=color, 
					width=0.3)

		plt.xticks(np.arange(len(group_names)), group_names)
		yticks = np.arange(-600, 800, 100)
		ytick_labels = [str(np.abs(y)) for y in yticks]
		plt.yticks(yticks, ytick_labels)
		plt.ylim(-150, 500)
		plt.xlim(-0.5, 3.5)
		plt.axhline(0, c='black', lw=1, zorder=0)
		plt.ylabel("# of genes w/\nincreased or decreased PTR")
		plt.title(f"Genes with increased/decreased \nExpression PTR "
				  f"(±{self.delta_ge}) after correction",
				 pad=10, fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)

	def plot_chrom_examples(self, save_dir=None):

		from src.utils import load_orf_data
		
		chrom_normalized_raw_rep1 = load_orf_data('output/copy_correction/chromatin/normalized_raw_rep1_shared.csv')
		chrom_cor_rep1 = load_orf_data('output/copy_correction/chromatin/normalized_corrected_rep1_shared.csv')

		chrom_normalized_raw_rep2 = load_orf_data('output/copy_correction/chromatin/normalized_raw_rep2_shared.csv')
		chrom_cor_rep2 = load_orf_data('output/copy_correction/chromatin/normalized_corrected_rep2_shared.csv')

		from src.Figure3_Copy_Correction import plot_orf_correction
		from src.sgd import get_gene_name_orf_name

		for orf in self.selected_chrom_genes:
			orf_name, gene_name = get_gene_name_orf_name(orf)
			plot_orf_correction(orf, 
				chrom_normalized_raw_rep1, chrom_cor_rep1, 
				chrom_normalized_raw_rep2, chrom_cor_rep2,
							   ylim=(0, 2500), ylabel="Chromatin window occupancy")
			save_figure_for_paper(f'{save_dir}/Copy_Correction_Chromatin_Example_{gene_name}.png')

	def plot_selected_chrom_examples_ptr_decrease(self):
		from src.sgd import get_gene_name_orf_name
		from src.figure_configs import FiguresConfig
		from src.utils import load_orf_data

		def plot_gene_examples(genes, colors, titles, suptitle):
			
			def plot_genes_corrections(replicate):
				chrom_normalized_raw = load_orf_data(f'output/copy_correction/chromatin/normalized_raw_rep{replicate}_shared.csv')
				chrom_cor = load_orf_data(f'output/copy_correction/chromatin/normalized_corrected_rep{replicate}_shared.csv')

				def plot_gene_correction(gene_name, color, title):
					orf_name, gene_name = get_gene_name_orf_name(gene_name)
					raw = chrom_normalized_raw.loc[orf_name]
					corrected = chrom_cor.loc[orf_name]

					plt.plot(raw, color=color, lw=2, label=f"{gene_name}, {title}")
					plt.plot(corrected, color=color, ls='dashed', lw=1)

				for i in range(len(genes)):
					plot_gene_correction(genes[i], colors[i], titles[i])

				from src.config import load_configs_by_config_type
				config1, config2 = load_configs_by_config_type('shared')
				config = config1 if replicate == 1  else config2
				(s_start, s_end), (second_s_start, second_s_end) = config.get_raw_s_tps()
				plt.axvline(s_start, c='black', lw=1, ls='dotted')
				plt.axvline(s_end, c='black', lw=1, ls='dotted')
				plt.axvline(second_s_start, c='black', lw=1, ls='dotted')
				plt.axvline(second_s_end, c='black', lw=1, ls='dotted')

				plt.legend()
				plt.ylim(0, 2300)

			plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT_WIDE)
			plt.subplot(1, 2, 1)
			plot_genes_corrections(1)
			plt.title("Replicate 1", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
			plt.xlabel("Time, min")
			plt.ylabel("Chromatin occupancy")

			plt.subplot(1, 2, 2)
			plot_genes_corrections(2)
			plt.title("Replicate 2", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
			plt.xlabel("Time, min")

			plt.suptitle(suptitle, 
						 fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
			plt.subplots_adjust(top=0.84)

		suptitle = "Decreased PTR following correction"
		genes = ['HSP60', 'CLB3']
		colors = [plt.get_cmap('RdBu_r')(0.9), plt.get_cmap('RdBu_r')(0.1)]
		titles = ['Early', 'Late']
		plot_gene_examples(genes, colors, titles, suptitle)


	def plot_ge_examples(self, orfs=None, save_dir=None):
		from src.config import read_yl_vst_data_rep

		ge_data_rep1 = read_yl_vst_data_rep(1)
		ge_data_rep2 = read_yl_vst_data_rep(2)

		ge_cor_rep1 = pd.read_csv('output/copy_correction/gene_expression/normalized_corrected_expression_rep1_shared.csv').set_index('orf_name')
		ge_cor_rep2 = pd.read_csv('output/copy_correction/gene_expression/normalized_corrected_expression_rep2_shared.csv').set_index('orf_name')
		ge_cor_rep1.columns = ge_data_rep1.columns
		ge_cor_rep2.columns = ge_data_rep2.columns

		if orfs is None:
			orfs = self.selected_tx_genes

		from src.sgd import get_gene_name_orf_name

		for orf in orfs:
			plot_orf_correction(orf, ge_data_rep1, ge_cor_rep1, ge_data_rep2, ge_cor_rep2)

			if save_dir is not None:
				orf_name, gene_name = get_gene_name_orf_name(orf)
				save_figure_for_paper(f'{save_dir}/Copy_Correction_Expression_Example_{gene_name}.png')


	def compute_gene_expression_ptrs(self):
		from src.gene_expression_deconv_analysis import GeneExpressionAnalysis

		# Let's examine the PTRs for the gene expression to get an idea of the
		# cell cycle regulated gene terms
		gene_expression_dir = 'output/deconvolve_sharedg1_0066_cc_2024_06_13/gene_expression/'
		gene_expression_a = GeneExpressionAnalysis(gene_expression_dir)
		gene_expression_a.compute_ptrs()
		plt.figure(figsize=(4, 3))
		plt.hist(gene_expression_a.gene_ptrs_df['ptr'], bins=100)
		plt.ylim(0, 200)
		thresh = 1.15
		plt.axvline(thresh, c='red', alpha=0.5)
		plt.title("Distribution of deconvolved gene expression PTRs")
		self.expression_ptr_threshold = thresh
		self.gene_expression_analysis = gene_expression_a

	def perform_gene_expression_go(self):
		from src.gene_ontology import GeneOntology
		gene_ontology = GeneOntology()
		sgd_rows = gene_ontology.orfs_with_go
		highest_cycling_expression = self.gene_expression_analysis.gene_ptrs_df[\
			self.gene_expression_analysis.gene_ptrs_df.ptr > self.expression_ptr_threshold]
		selected_genes = highest_cycling_expression
		selected_genes = selected_genes.join(sgd_rows[['name']])
		selected_orfs = selected_genes['name'].values
		gene_ontology.run_go(selected_orfs)
		self.gene_ontology = gene_ontology

	def plot_gene_ontology_violins(self, directory):

		from src.gene_ontology import genes_for_go

		for _, go_row in self.gene_ontology.results_sig_df.iterrows():
			go_term = go_row['name']
			go_id = go_row['id']

			# Get the example gene orfs for plotting
			example_genes, _ = genes_for_go(self.gene_ontology.orfs_with_go, go_id)
			example_orfs = example_genes.index

			from scipy.stats.distributions import norm

			self.plot_violin_ptr_11()

			# --------

			vp = self.violin_plotter
			selected_df = vp.dfs_to_plot[0].loc[example_orfs]

			x, y = selected_df.group_id, selected_df.distance_11

			x = x + norm.rvs(loc=0, scale=0.01, size=len(x))
			plt.scatter(x, y, s=20, zorder=100, edgecolor='white', facecolor='green')
			
			go_name = go_term.title().replace("Dna", "DNA")
			go_term_title = (f"{go_id}, n={len(example_orfs)}\n{go_name}")

			plt.title(go_term_title)

			go_save_title = go_term.replace(' ', '_')[0:13]
			save_path = f'{directory}/copy_chrom_go_{go_save_title}.png'
			save_figure_for_paper(save_path)
			plt.close()
			print("Wrote to " + save_path)

	def plot_violin_plot_selected_GO_terms(self):

		from src.gene_ontology import genes_for_go
		from scipy.stats.distributions import norm

		go_terms = ['GO:0000079',
					'GO:0003697']
		go_names = [
			'Regulation of Cyclin dependent S/T kinase activity',
			'Single stranded DNA binding'
		]

		selected_genes = self.selected_chrom_genes

		colors = ['green', 'purple']
		x_offset = [-0.05, 0.05]

		self.plot_violin_ptr_11()

		vp = self.violin_plotter
		dfs_to_plot = vp.dfs_to_plot[0].copy()
		x, y = dfs_to_plot.group_id, dfs_to_plot.distance_11

		np.random.seed(123)

		dfs_to_plot['x'] = x = x + \
			norm.rvs(loc=0, scale=0.01, size=len(x))

		import matplotlib.patheffects as path_effects

		for i in range(len(go_terms)):
			
			go_term = go_terms[i]
			selected_go_orfs = genes_for_go(self.gene_ontology.orfs_with_go, 
				go_term)[0].index
			color = colors[i]

			selected_df = dfs_to_plot.loc[selected_go_orfs]
			x, y = selected_df['x'] + x_offset[i], selected_df.distance_11

			label = f"{go_names[i]}, n={len(x)}"
			
			plt.scatter(x, y, 
				s=20, zorder=50, edgecolor='white', facecolor=color,
					   label=label, lw=1, marker='o')

			for gene in selected_genes:
				from src.sgd import get_gene_name_orf_name
				orf_name, gene_name = get_gene_name_orf_name(gene)
				if orf_name in selected_go_orfs:
					row = selected_df.loc[orf_name]

					ha = 'left'
					x = row.x + x_offset[i]*2.

					if x_offset[i] < 0: ha = 'right'
					else: ha = 'left'

					text = plt.text(x, row.distance_11, gene_name, 
						ha=ha, zorder=100)
					text.set_path_effects([path_effects.Stroke(linewidth=3., 
						foreground='white'), path_effects.Normal()])

		plt.yticks(np.arange(-0.04, 0.05, 0.02))
		plt.ylim(-0.05, 0.04)
		plt.legend(loc='lower right')

		plt.title("Copy correction PTR adjustment")


	def plot_gene_expression_ptr_vs_diff_11(self):
		ge_distance_data = self.mean_chrom_ptrs_w_distances_11.join(self.gene_expression_analysis.gene_ptrs_df)
		ge_distance_data = ge_distance_data.join(self.mean_chrom_ptrs['replication_time'])

		plt.scatter(ge_distance_data.ptr, ge_distance_data.distance_11, s=1,
				   c=ge_distance_data.replication_time, cmap='RdBu_r',
				   vmin=6, vmax=15)
		plt.colorbar()
		plt.axhline(0, c='black', ls='dotted', lw=1)
		plt.title("Gene expression PTR compared to copy correction change")


	def plot_heatmap_raw_corrected_mnase(self, replicate):
		"""Plot the heatmap of the raw and corrected MNase occupancy data"""

		from src.utils import load_orf_data

		normalized_raw = load_orf_data(f'output/copy_correction/chromatin/normalized_raw_rep{replicate}_shared.csv')
		chrom_cor = load_orf_data(f'output/copy_correction/chromatin/normalized_corrected_rep{replicate}_shared.csv')
		
		#normalized_raw = chrom_raw / chrom_raw.sum(axis=0).values.reshape((1, -1))
		#normalized_raw *= chrom_cor.sum(axis=0).iloc[0]

		if replicate == 1:
			orfs_sorted = self.ge_ptrs_rep1.sort_values('replication_time').index
		else:
			orfs_sorted = self.ge_ptrs_rep2.sort_values('replication_time').index
		raw = normalized_raw.loc[orfs_sorted]
		corrected = chrom_cor.loc[orfs_sorted]
		difference = corrected-raw

		from src.global_config import GlobalConstants
		from src.config import load_configs_by_config_type

		config1, config2 = load_configs_by_config_type('shared')
		if replicate == 1: config = config1
		else: config = config2
		(s_start, s_end), (second_s_start, second_s_end) = config.get_raw_s_tps()

		def plot_im(dat, vmin=0, vmax=4000, cmap='viridis'):
			# plt.imshow(dat, aspect='auto', interpolation='none',
			# 	extent=[-5, GlobalConstants.CHROM_WT1_TIMEPOINTS[-1]+5, 
			# 			0, 100],
			# 		  vmin=vmin, vmax=vmax, cmap=cmap)

			# print(dat.shape)
			# first_col = dat.values[:, 0]
			# logfold_dat = np.log2(dat / (first_col.reshape((-1, 1))))

			plt.imshow(dat, aspect='auto', interpolation='none',
				extent=[-5, GlobalConstants.CHROM_WT1_TIMEPOINTS[-1]+5, 
						0, 100],
					  vmin=1000, vmax=2500, cmap='viridis')

			plt.yticks([])
			plt.colorbar()

		plt.figure(figsize=FiguresConfig.FIGSIZE_WIDE)
		plt.subplot(1, 3, 1)
		plot_im(raw)
		# plt.ylabel("Genes sorted by replication time")
		# plt.xlabel("Time, min")

		plt.title("Raw", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)

		plt.subplot(1, 3, 2)
		plot_im(corrected)
		plt.title("Copy corrected", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)

		# plt.subplot(1, 3, 3)
		# plot_im(difference, vmin=-100, vmax=100, cmap='RdBu_r')
		# plt.title("Difference", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)

		# plt.suptitle(f"Gene 1kb MNase-seq occupancy, replicate {replicate}", 
		# 	fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		# plt.subplots_adjust(top=0.87, wspace=0.2)


	def select_genes_for_go_repl(self, go_id, timing_group=None, diff_cutoffs=[-1, 1]):
		from src.gene_ontology import genes_for_go
		selected_orf_ids, _ = genes_for_go(self.gene_ontology.orfs_with_go, go_id)
		genes = self.gene_ontology.orfs_with_go
		ptrs_w_dist = self.mean_chrom_ptrs_w_distances_11.loc[selected_orf_ids.index]
		selected = ptrs_w_dist[(ptrs_w_dist.distance_11 > diff_cutoffs[0]) & 
						(ptrs_w_dist.distance_11 < diff_cutoffs[1])].join(genes[['name']])
		if timing_group is not None:
			selected = selected[(selected.group_name == timing_group)]
		return selected

def rep_quantile_colors():
	group_colors = []
	for color_prop in [0, 0.33, 0.66, 1.]:
		prop = color_prop*0.94+0.03
		color = plt.get_cmap('Spectral')(prop)

		# Increase lightness for readability
		color = adjust_lightness_saturation(color, 1.1, 0.8)

		group_colors.append(color)
	return group_colors


def get_color_for_rep_group(key):

	colors = rep_quantile_colors()
	mapping = {
		'Early': colors[0],
		'Early-Mid': colors[1],
		'Mid-Late': colors[2],
		'Late': colors[3]
	}
	return mapping[key]


def ptr_cmap():
	from src.plot_helpers import adjust_lightness_saturation_colormap

	cmap = plt.get_cmap('inferno')
	#cmap = adjust_lightness_saturation_colormap(cmap, 1., 0.8, 'SatSpectral')
	return cmap

def _plot_ann_text(x, y, text, fontsize=16, 
	ha='left', va='bottom', zorder=1, 
	textcolor='black', bordercolor='white'):
	"""
	Can't plot this to ax for some reason, revisit this later, check cd 
	paper code for plotting arbitrary text
	"""

	import matplotlib.patheffects as patheffects

	plt.text(x, y, text, fontsize=fontsize, weight='normal', style='italic', 
		path_effects=[patheffects.withStroke(linewidth=2, foreground=bordercolor)],
		color=textcolor, va=va, ha=ha, zorder=zorder)


def plot_selected_genes(plot_data, selected_genes, keyx, keyy, xlim, ylim):
	"""Plot annotations on scatter"""
	from src.sgd import get_gene_name_orf_name
	from src.sgd import get_gene_title_name

	for gene_name in selected_genes:

		orf_name, gene_name = get_gene_name_orf_name(gene_name)
		gene_title = gene_name

		selected_plot_data = plot_data.loc[orf_name]
		x, y = selected_plot_data[keyx], selected_plot_data[keyy]

		if y > ylim[1] or x > xlim[1]:
			continue

		x_text = x
		if x <= y:
			va = 'bottom'
			y_text = y+0.005
			ha='right'
		else:
			va = 'top'
			y_text = y-0.005
			ha='left'

		custom_gene_positioning = {
			'APA1': {
				'va':'center',
				'ha':'left',
				'y': y
			}
		}

		if gene_name in custom_gene_positioning.keys():
			ha = custom_gene_positioning[gene_name]['ha']
			va = custom_gene_positioning[gene_name]['va']
			y_text = custom_gene_positioning[gene_name]['y']
			x_text = x + 0.007

		color = get_color_for_rep_group(selected_plot_data.group_name)
		_plot_ann_text(x_text, y_text, gene_title, zorder=21, fontsize=11, ha=ha, va=va,
			bordercolor='white')
		plt.scatter(x, y, s=30, edgecolor=color, facecolors='none', zorder=3, marker='D')


def plot_orf_correction(gororf, gene_data_rep1, corrected_data_rep1, gene_data_rep2, corrected_data_rep2,
	ylim=(0, 18), ylabel='Expression, VST'):
	"""Plot time course of corrected and uncorrected genes"""

	from src.sgd import get_gene_name_orf_name
	from src.config import load_configs_by_config_type

	config1, config2 = load_configs_by_config_type('shared')

	plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT_WIDE)

	plt.subplot(1, 2, 1)
	(s_start, s_end), (second_s_start, second_s_end) = config1.get_raw_s_tps()

	orf_name, gene_name = get_gene_name_orf_name(gororf)
	color = plt.get_cmap('Spectral')(0.1)
	raw_color = plt.get_cmap('Spectral')(0.93)
	plt.plot(gene_data_rep1.loc[orf_name], c=raw_color, lw=3, label="Raw")
	plt.plot(corrected_data_rep1.loc[orf_name], c=color, ls='dashed', lw=2, label="Corrected")
	plt.ylim(*ylim)
	
	from src.sgd import get_gene_title_name
	plt.suptitle(get_gene_title_name(orf_name))
	plt.legend()
	plt.xlabel("Time, min")
	plt.ylabel(ylabel)
	plt.title("Replicate 1", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
	plt.axvline(s_start, c='black', ls='dotted', lw=1)
	plt.axvline(s_end, c='black', ls='dotted', lw=1)

	plt.axvline(second_s_start, c='black', ls='dotted', lw=1)
	plt.axvline(second_s_end, c='black', ls='dotted', lw=1)


	plt.subplot(1, 2, 2)

	(s_start, s_end), (second_s_start, second_s_end) = config2.get_raw_s_tps()
	plt.plot(gene_data_rep2.loc[orf_name], c=raw_color, lw=3, label="Raw")
	plt.plot(corrected_data_rep2.loc[orf_name], c=color, ls='dashed', lw=2, label="Corrected")
	plt.ylim(*ylim)
	plt.title("Replicate 2", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
	
	from src.sgd import get_gene_title_name
	plt.suptitle(get_gene_title_name(orf_name), fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
	plt.legend()
	plt.xlabel("Time, min")
	plt.subplots_adjust(top=0.8)

	plt.axvline(s_start, c='black', ls='dotted', lw=1)
	plt.axvline(s_end, c='black', ls='dotted', lw=1)

	plt.axvline(second_s_start, c='black', ls='dotted', lw=1)
	plt.axvline(second_s_end, c='black', ls='dotted', lw=1)

