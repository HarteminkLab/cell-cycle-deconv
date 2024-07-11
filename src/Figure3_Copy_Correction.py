
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure_configs import FiguresConfig
from src.figure_configs import save_figure_for_paper


class Figure3CopyCorrection(object):
	"""Load and plot figures for the third result figure"""

	def __init__(self):

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
		self.selected_chrom_genes = ['CDC28', 'RPA34', # Increase
									 'MRE11', 'AFR1', 'DUN1', # Decrease
									 'SSK22' # Control genes
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

		alphas = origins[eff_key].values
		alphas[alphas > 1.] = 1
		alphas[alphas < 0] = 0
		plt.axhline(1, c='black', lw=1.5)
		for i in range(len(ys)):
			tp = replication_tps[i]
			y = ys[i]
			alpha = alphas[i]*1#0.5
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
		origins_chr['alpha'] = origins_chr[eff_key] + 0.2
		origins_chr.loc[origins_chr['alpha'] < 0.2, 'alpha'] = 0.2 # Span values from 0.2-1.0, original values are from 0-0.75 (with -1's as not found)

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

	def plot_chrom_ptr_correction(self, selected_genes=None):
		
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
		plt.title(f"Mean Chromatin PTR,\nRaw vs Copy # Corrected n={n}", 
			fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.plot([1, 2], [1, 2], c='black', lw=0.5, ls='dotted', zorder=10)

		xlim, ylim = (1, 1.5), (1, 1.5)
		plt.xlim(*xlim)
		plt.ylim(*ylim)

		if selected_genes is None:
			selected_genes = self.selected_chrom_genes

		plot_selected_genes(plot_data, selected_genes, 'normalized_raw_ptr', 'normalized_corrected_ptr',
			xlim, ylim)
		
		plt.xlabel("Raw expression PTR")
		plt.ylabel("Copy-number-corrected expression PTR")

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
		
		def load_orf_data(path):
			dat = pd.read_csv(path).set_index('orf_name')
			dat.columns = dat.columns.astype(int)
			return dat

		chrom_raw_rep1 = load_orf_data('output/copy_correction/chromatin/raw_sums_rep1_shared.csv')
		chrom_cor_rep1 = load_orf_data('output/copy_correction/chromatin/normalized_corrected_rep1_shared.csv')

		chrom_raw_rep2 = load_orf_data('output/copy_correction/chromatin/raw_sums_rep1_shared.csv')
		chrom_cor_rep2 = load_orf_data('output/copy_correction/chromatin/normalized_corrected_rep1_shared.csv')

		from src.Figure3_Copy_Correction import plot_orf_correction
		from src.sgd import get_gene_name_orf_name

		for orf in self.selected_chrom_genes:
			orf_name, gene_name = get_gene_name_orf_name(orf)
			plot_orf_correction(orf, chrom_raw_rep1, chrom_cor_rep1, chrom_raw_rep2, chrom_cor_rep2,
							   ylim=(0, 2500), ylabel="Chromatin window occupancy")
			save_figure_for_paper(f'{save_dir}/Copy_Correction_Chromatin_Example_{gene_name}.png')


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


from src.plot_helpers import adjust_lightness_saturation

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

	cmap = plt.get_cmap('RdBu')
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

		_plot_ann_text(x, y+0.005, gene_title, zorder=21, fontsize=8, ha='center')
		color = get_color_for_rep_group(selected_plot_data.group_name)
		plt.scatter(x, y, s=30, edgecolor=color, facecolors='none', zorder=3, marker='D')


def plot_orf_correction(gororf, gene_data_rep1, corrected_data_rep1, gene_data_rep2, corrected_data_rep2,
	ylim=(0, 18), ylabel='Expression, VST'):
	"""Plot time course of corrected and uncorrected genes"""
	from src.sgd import get_gene_name_orf_name

	plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT_WIDE)

	plt.subplot(1, 2, 1)

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

	plt.subplot(1, 2, 2)
	plt.plot(gene_data_rep2.loc[orf_name], c=raw_color, lw=3, label="Raw")
	plt.plot(corrected_data_rep2.loc[orf_name], c=color, ls='dashed', lw=2, label="Corrected")
	plt.ylim(*ylim)
	plt.title("Replicate 2", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
	
	from src.sgd import get_gene_title_name
	plt.suptitle(get_gene_title_name(orf_name), fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
	plt.legend()
	plt.xlabel("Time, min")
	plt.subplots_adjust(top=0.8)
