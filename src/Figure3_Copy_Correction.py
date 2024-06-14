
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure_configs import FiguresConfig

class Figure3CopyCorrection(object):
	"""Load and plot figures for the third result figure"""

	def __init__(self):

		self.chrom_ptrs_rep1 = pd.read_csv('output/copy_correction/chromatin/ptrs_rep1_shared.csv')\
			.set_index('orf_name')
		self.chrom_ptrs_rep2 = pd.read_csv('output/copy_correction/chromatin/ptrs_rep2_shared.csv')\
			.set_index('orf_name')
		self.mean_chrom_ptrs = (self.chrom_ptrs_rep1 + self.chrom_ptrs_rep2) / 2.

		self.ge_ptrs_rep1 = pd.read_csv('output/copy_correction/gene_expression/expression_ptr_comparison_rep1_shared.csv')\
			.set_index('orf_name')
		self.ge_ptrs_rep2 = pd.read_csv('output/copy_correction/gene_expression/expression_ptr_comparison_rep2_shared.csv')\
			.set_index('orf_name')
		self.mean_ge_ptrs = (self.ge_ptrs_rep1 + self.ge_ptrs_rep2) / 2.

	def plot_chrom_ptr_correction(self):
		
		plot_data = self.mean_chrom_ptrs
		plt.figure(figsize=FiguresConfig.FIGSIZE_SQUARE_WIDE)
		plt.scatter(plot_data.normalized_raw_ptr, 
					plot_data.normalized_corrected_ptr,
					facecolors='none', lw=2, edgecolor='#ccc', s=2, zorder=1)
		plt.scatter(plot_data.normalized_raw_ptr, 
					plot_data.normalized_corrected_ptr,
					c=plot_data.replication_time,
					vmax=15,
					s=1, cmap='Spectral', zorder=2)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Replication time, min", rotation=270, va='bottom')

		n = len(plot_data)
		plt.title(f"Mean Chromatin PTR,\nRaw vs Copy # Corrected n={n}", 
			fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.plot([1, 2], [1, 2], c='black', lw=0.5, ls='dotted', zorder=0)
		plt.xlim(1, 1.5)
		plt.ylim(1, 1.5)
		
		plt.xlabel("Raw expression PTR")
		plt.ylabel("Copy-number-corrected expression PTR")

	def plot_ge_ptr_correction(self):
		
		plot_data = self.mean_ge_ptrs
		plt.figure(figsize=FiguresConfig.FIGSIZE_SQUARE_WIDE)
		plt.scatter(plot_data.raw_ptr, 
					plot_data.corrected_ptr,
					facecolors='none', lw=2, edgecolor='#ccc', s=2, zorder=1)
		plt.scatter(plot_data.raw_ptr, 
					plot_data.corrected_ptr,
					c=plot_data.replication_time,
					vmax=15,
					s=1, cmap='Spectral', zorder=2)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Replication time, min", rotation=270, va='bottom')

		n = len(plot_data)
		plt.title(f"Gene expression PTR,\nRaw vs Copy # Corrected n={n}", 
			fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.plot([0, 2], [0, 2], c='black', lw=0.5, ls='dotted', zorder=0)

		plt.xticks(np.arange(1, 1.25, 0.05))
		plt.yticks(np.arange(1, 1.25, 0.05))

		plt.xlim(0.99, 1.2)
		plt.ylim(0.99, 1.2)
		
		plt.xlabel("Raw expression PTR")
		plt.ylabel("Copy-number-corrected expression PTR")

	def plot_chrom_ptr_rep_quantiles(self):
		from src.helpers import get_quantile_values

		chrom_ptrs = self.mean_chrom_ptrs.copy()
		n = len(chrom_ptrs)

		log_ratio = np.log2(chrom_ptrs['normalized_corrected_ptr'] / chrom_ptrs['normalized_raw_ptr']+1)
		chrom_ptrs['log_ratio'] = log_ratio

		segments, qvals, lens = get_quantile_values(chrom_ptrs.replication_time, 
			q=[0.25, 0.5, 0.75])

		early_genes, early_mid_genes, mid_late_genes, late_genes = segments

		from src.boxplot import BoxPlotPlotter

		chrom_ptrs.loc[early_genes.index, 'group_id'] = 0
		chrom_ptrs.loc[early_mid_genes.index, 'group_id'] = 1
		chrom_ptrs.loc[mid_late_genes.index, 'group_id'] = 2
		chrom_ptrs.loc[late_genes.index, 'group_id'] = 3

		chrom_ptrs.loc[early_genes.index, 'group_name'] = 'Early'
		chrom_ptrs.loc[early_mid_genes.index, 'group_name'] = 'Early-Mid'
		chrom_ptrs.loc[mid_late_genes.index, 'group_name'] = 'Mid-Late'
		chrom_ptrs.loc[late_genes.index, 'group_name'] = 'Late'

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
		for color_prop in [0, 0.33, 0.66, 1.]:
			prop = color_prop*0.94+0.03
			color = plt.get_cmap('Spectral')(prop)
			box_plotter.group_colors.append(color)

		fig = plt.figure(figsize=FiguresConfig.FIGSIZE_SHORT)
		box_plotter.plot_box_plot(ax=plt.gca(), title='')
		plt.ylabel("Log2-ratio change in PTR")
		plt.xlabel("Replication timing")
		plt.title(f"Chromatin copy correction\nchange in PTR, n={n}", 
				  fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.axhline(1, c='black', lw=0.5, ls='dotted', zorder=0)

		plt.ylim(-0.07, 0.07)
		plt.ylim(0.95, 1.05)


	def plot_ge_ptr_rep_quantiles(self):
		from src.helpers import get_quantile_values

		chrom_ptrs = self.mean_ge_ptrs.copy()
		n = len(chrom_ptrs)

		log_ratio = np.log2(chrom_ptrs['corrected_ptr'] / chrom_ptrs['raw_ptr']+1)
		chrom_ptrs['log_ratio'] = log_ratio

		segments, qvals, lens = get_quantile_values(chrom_ptrs.replication_time, 
			q=[0.25, 0.5, 0.75])

		early_genes, early_mid_genes, mid_late_genes, late_genes = segments

		from src.boxplot import BoxPlotPlotter

		chrom_ptrs.loc[early_genes.index, 'group_id'] = 0
		chrom_ptrs.loc[early_mid_genes.index, 'group_id'] = 1
		chrom_ptrs.loc[mid_late_genes.index, 'group_id'] = 2
		chrom_ptrs.loc[late_genes.index, 'group_id'] = 3

		chrom_ptrs.loc[early_genes.index, 'group_name'] = 'Early'
		chrom_ptrs.loc[early_mid_genes.index, 'group_name'] = 'Early-Mid'
		chrom_ptrs.loc[mid_late_genes.index, 'group_name'] = 'Mid-Late'
		chrom_ptrs.loc[late_genes.index, 'group_name'] = 'Late'

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
		plt.xlabel("Replication timing")
		plt.title(f"Gene expression copy correction\nchange in PTR, n={n}", 
				  fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=10)
		plt.axhline(1, c='black', lw=0.5, ls='dotted', zorder=0)

		plt.ylim(-0.07, 0.07)
		plt.ylim(0.97, 1.05)


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