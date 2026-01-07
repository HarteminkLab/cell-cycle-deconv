import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import load_default_chrom_configs
from src.utils import print_fl, mkdir_safe
from src.figure_configs import save_figure_for_paper
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import ListedColormap
from src.sgd import get_gene_title_name
from src.plot_helpers import plot_trajectory_deconvolved_values


trajectory_lims_mapping = {
	'expression': (-1, 17),
	'promoter_occupancy': (-0.25, 3),
	'nucleosome_occupancy': (-0.5, 6),
	'nucleosome_entropy': (-0.5, 6),
}

class IntegratedChromatinExpressionAnalyzer:
	"""
	Integrates chromatin and expression analysis to perform combined analyses
	and visualizations.
	
	This class coordinates between ChromatinMetricsProcessor and 
	ExpressionAnalysisProcessor to:
	- Compare chromatin metrics with expression data
	- Analyze correlations between chromatin and expression PTRs
	- Create phase-state plots showing chromatin vs expression by cell cycle phase
	- Identify genes with coordinated chromatin-expression changes
	"""
	
	def __init__(self, chromatin_processor, expression_processor, output_dir: str):
		"""
		Initialize the integrated analyzer.
		
		Parameters
		----------
		chromatin_processor : ChromatinMetricsProcessor
			Instance with computed chromatin metrics and PTRs
		expression_processor : ExpressionAnalysisProcessor  
			Instance with computed expression data and PTRs
		output_dir : str
			Directory for saving output files and figures
		"""
		self.chromatin_processor = chromatin_processor
		self.expression_processor = expression_processor
		self.output_dir = output_dir
		
		# Load configuration for phase analysis
		self.config = chromatin_processor.config1
		
		# Initialize storage for analysis results
		self.correlation_results = {}
		self.coordinated_genes = {}
		self.phase_analysis_results = {}
		
	
	def compute_ptr_correlations(self, chromatin_data_source='deconvolved',
		set_mode='genic'):
		"""
		Compute correlations between chromatin PTRs and expression PTRs.
		
		Parameters
		----------
		chromatin_data_source : str, optional
			Which chromatin data to use: 'deconvolved', 'raw_rep1', or 'raw_rep2'
			
		Returns
		-------
		dict
			Dictionary with correlation results for each chromatin metric
		"""
		
		# Get chromatin PTRs based on data source
		if chromatin_data_source == 'deconvolved':
			chromatin_ptrs = self.chromatin_processor.normalized_ptr_deconvolved
		elif chromatin_data_source == 'raw_rep1':
			chromatin_ptrs = self.chromatin_processor.normalized_ptr_rep1
		elif chromatin_data_source == 'raw_rep2':
			chromatin_ptrs = self.chromatin_processor.normalized_ptr_rep2
		else:
			raise ValueError(f"Unknown chromatin data source: {chromatin_data_source}")
		
		# Get expression PTRs (filtered to genic transcripts)
		expression_ptrs = self.expression_processor.get_genic_ptrs()
		
		# Compute correlations for each chromatin metric
		correlation_results = {}
		
		for metric_name in ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']:
			metric_ptrs = chromatin_ptrs[metric_name]

			joined_ptrs = metric_ptrs.join(expression_ptrs[['ptr']], how='inner')
			joined_ptrs.columns = ['chromatin_ptr', 'expression_ptr']
			
			# Remove NaN values
			joined_ptrs = joined_ptrs.fillna(1.0)
			
			# Compute correlations
			pearson_r, pearson_p = pearsonr(joined_ptrs.chromatin_ptr, joined_ptrs.expression_ptr)
			spearman_r, spearman_p = spearmanr(joined_ptrs.chromatin_ptr, joined_ptrs.expression_ptr)
			
			correlation_results[metric_name] = {
				'n_genes': len(joined_ptrs),
				'pearson_r': pearson_r,
				'pearson_p': pearson_p,
				'spearman_r': spearman_r,
				'spearman_p': spearman_p,
				'joined_data': joined_ptrs
			}
			
			print_fl(f"{metric_name}: n={len(joined_ptrs)}, "
					f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), "
					f"Spearman r={spearman_r:.3f} (p={spearman_p:.2e})")
		
		self.correlation_results[chromatin_data_source] = correlation_results
		return correlation_results
	
	def plot_ptr_correlations(self, chromatin_data_source='deconvolved', 
							  color_by='density',
							  figsize=(8, 3.5), save_plots=False):
		"""
		Plot correlations between chromatin PTRs and expression PTRs.
		
		Parameters
		----------
		chromatin_data_source : str, optional
			Which chromatin data to use for plotting
		figsize : tuple, optional
			Figure size
		save_plots : bool, optional
			Whether to save plots to disk
		"""
		if chromatin_data_source not in self.correlation_results:
			self.compute_ptr_correlations(chromatin_data_source)
		
		results = self.correlation_results[chromatin_data_source]
		
		fig, axes = plt.subplots(1, 3, figsize=figsize)
		
		metric_names = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		titles = ['Promoter occupancy', 'Nucleosome entropy', 'Nucleosome occupancy']
		
		for i, (metric_name, title) in enumerate(zip(metric_names, titles)):
			if metric_name not in results:
				continue
				
			ax = axes[i]
			results_for_metric = results[metric_name]
			data = results_for_metric['joined_data']

			from src.DensityScatterPlotter import DensityScatterPlotter

			from pipeline.chromatin_metrics_processor import plot_formatting_map
			cmap = plot_formatting_map[metric_name]['cmap']
			chrom_bw = plot_formatting_map[metric_name]['bw']
			ptr_lims = plot_formatting_map[metric_name]['ptr_lims']

			if color_by == 'density':
				dsc_plotter = DensityScatterPlotter()
				dsc_plotter.plot_outline = True
				dsc_plotter.outline_color = '#eee'
				dsc_plotter.logz=True

				# Plot the PTRs
				dsc_plotter.set_data(data.chromatin_ptr, data.expression_ptr)
				dsc_plotter.bw = chrom_bw, 0.25
				dsc_plotter.cmap = cmap
				dsc_plotter.s = 5
				dsc_plotter.plot_ax(ax)

			elif color_by in self.trajectory_area_linkages[metric_name].columns:

				#if color_by == 'trajectory_area' or 
				vmin = 0
				vmax = 0.5

				# We only have area plots for the deconvolved data, currently
				if not chromatin_data_source == 'deconvolved':
					raise ValueError("Unimplemented trajectory area plotting for non deconvolved data set")

				data = self.trajectory_area_linkages[metric_name]
				color_by_value = data[color_by]

				ax.scatter(data.chromatin_ptr, data.expression_ptr, edgecolor='#aaa',
					facecolor='none',
					s=7) # outline
				ax.scatter(data.chromatin_ptr, data.expression_ptr, c=color_by_value,
					cmap=cmap, s=5, vmax=vmax, vmin=vmin)
			else:
				raise ValueError("Error in color_by argument: " + color_by)
			
			# Labels and title
			ax.set_xlabel(f'{title} PTR', fontsize=13)
			ax.set_ylabel('Expression PTR', fontsize=13)
			ax.set_title(f'{title}\nn={results_for_metric["n_genes"]}',
				fontsize=14)

			chromatin_threshold_value = self.quantile_threshold_values[metric_name]
			expression_threshold_value = self.quantile_threshold_values['expression']

			from scipy.stats import pearsonr, spearmanr

			# Compute correlation measures
			pearsonr_value, pearson_p = pearsonr(data.chromatin_ptr, data.expression_ptr)

			# ax.plot([chromatin_threshold_value, chromatin_threshold_value], 
			# 	[expression_threshold_value, 10], 
			# 	c='red', ls='solid', alpha=0.5, lw=1)
			# ax.plot([chromatin_threshold_value, 10], [expression_threshold_value, 
			# 	expression_threshold_value], 
			# 	c='red', ls='solid', alpha=0.5, lw=1)

			# num_chromatin_and_tx = len(data[(data.chromatin_ptr > chromatin_threshold_value) & 
			# 	(data.expression_ptr > expression_threshold_value)])

			x = (ptr_lims[1]-ptr_lims[0])*0.75 + ptr_lims[0]
			y = 5

			ax.text(x, y, f"$R^2$= {pearsonr_value:.2f}",
				fontsize=12, fontweight='regular', ha='center', va='top')

			# ax.text(x, y, f"{num_chromatin_and_tx} genes"\
			# 	f"\n{num_chromatin_and_tx/n*100:.1f}%",
			# 	fontsize=12, fontweight='regular', ha='center', va='top')

			ax.set_xlim(*ptr_lims)
			ax.set_ylim(0.7, 6)
		
		if chromatin_data_source == 'deconvolved':
			title = "Chromatin vs Expression"
		else:
			title = f'Chromatin vs Expression PTR concordance ({chromatin_data_source})'

		plt.suptitle(title, fontweight='demi', fontsize=15)
		plt.tight_layout()
		
		return fig
	
	def apply_threshold(self, chromatin_data_source='deconvolved',
								  quantile_ptr_threshold=0.95):
		"""
		Identify genes showing coordinated chromatin-expression changes.
		
		Parameters
		----------
		chromatin_data_source : str, optional
			Which chromatin data to use
		ptr_threshold : float, optional
			Minimum PTR value for both chromatin and expression
		correlation_threshold : float, optional
			Minimum correlation coefficient for coordinated genes
			
		Returns
		-------
		dict
			Dictionary with coordinated genes for each chromatin metric
		"""
		if chromatin_data_source not in self.correlation_results:
			self.compute_ptr_correlations(chromatin_data_source)

		self.quantile_ptr_threshold = quantile_ptr_threshold
		self.quantile_threshold_values = {}
		
		results = self.correlation_results[chromatin_data_source]
		coordinated_genes = {}
		
		for metric_name, data in results.items():

			chromatin_threshold_value = np.quantile(data['joined_data'].chromatin_ptr, q=quantile_ptr_threshold)
			expression_threshold_value = np.quantile(data['joined_data'].expression_ptr, q=quantile_ptr_threshold)

			self.quantile_threshold_values[metric_name] = chromatin_threshold_value
			self.quantile_threshold_values['expression'] = expression_threshold_value

			# Filter genes with high PTRs in both chromatin and expression
			high_chromatin_mask = data['joined_data'].chromatin_ptr > chromatin_threshold_value
			high_expression_mask = data['joined_data'].expression_ptr > expression_threshold_value
			high_both_mask = high_chromatin_mask & high_expression_mask
			
			coordinated_genes[metric_name] = {
				'genes': data['joined_data'].loc[high_both_mask].index,
				'masked_ptrs': data['joined_data'].loc[high_both_mask],
				'n_genes': np.sum(high_both_mask)
			}
		
		# Set coordinated genes list by applying both quantile thresholds
		self.coordinated_genes[chromatin_data_source] = coordinated_genes

	def plot_orf_phase_state_raw(self, orf_or_gene_name, chromatin_key=None, 
		replicate=1, xlim=(-0.5, 8), ylim=(-0.5, 12)):
		# Get chromatin PTRs based on data source

		if replicate == 1:
			chromatin_metrics_data = self.chromatin_processor.normalized_raw_rep1_metrics[chromatin_key]
			raw_transcription_data = self.expression_processor.raw_rep1_expression_data
		elif replicate == 2:
			chromatin_metrics_data = self.chromatin_processor.normalized_raw_rep2_metrics[chromatin_key]
			raw_transcription_data = self.expression_processor.raw_rep2_expression_data
		else:
			raise ValueError()

		from src.sgd import get_gene_name_orf_name, get_gene_title_name

		if orf_or_gene_name in chromatin_metrics_data.index:
			orf_name = orf_or_gene_name
		else:
			orf_name, gene_name = get_gene_name_orf_name(orf_or_gene_name)
		
		# Extract time course data
		chromatin_sample = chromatin_metrics_data.loc[orf_name]
		expression_sample = raw_transcription_data.loc[orf_name]

		if len(chromatin_sample) > len(expression_sample):
			# Select subset of chromatin sample to match expression size
			# todo: easiest method to match lengths, a more
			# sophisticated method may be needed, but this conveys the raw data
			# as a rough estimate for these plots
			chromatin_sample = chromatin_sample[:len(expression_sample)]

		plt.plot(chromatin_sample, expression_sample, color='#777', lw=1, zorder=1)

		from pipeline.chromatin_metrics_processor import plot_formatting_map

		cmap = plot_formatting_map[chromatin_key]['cmap']

		z = np.arange(len(chromatin_sample))

		plt.scatter(chromatin_sample, expression_sample, s=3, lw=2,
			edgecolor='#aaa', facecolor='none', zorder=2)
		plt.scatter(chromatin_sample, expression_sample, s=2, c=z, cmap=cmap, zorder=2)
		
		plt.ylabel(f"Expression")
		plt.xlim(*xlim)
		plt.ylim(*ylim)
		plt.title(chromatin_key)
	
	def plot_orf_phase_state_deconvolved(self, orf_or_gene_name, chromatin_key=None, 
		xlim=(-0.5, 8), ylim=(-0.5, 12), lw=5, plot_arrows=False, color_arrows=True):

		# Get chromatin PTRs based on data source
		chromatin_metrics_data = self.chromatin_processor.normalized_deconvolved_metrics

		from src.sgd import get_gene_name_orf_name, get_gene_title_name
		
		# Get data
		deconvolved_chromatin_data = chromatin_metrics_data[chromatin_key]
		deconvolved_transcription_data = self.expression_processor.expression_data
		
		if orf_or_gene_name in chromatin_metrics_data[chromatin_key].index:
			orf_name = orf_or_gene_name
		else:
			orf_name, gene_name = get_gene_name_orf_name(orf_or_gene_name)
		
		# Extract time course data
		chromatin_sample = deconvolved_chromatin_data.loc[orf_name]
		expression_sample = deconvolved_transcription_data.loc[orf_name]

		return plot_trajectory_deconvolved_values(self.config, 
			chromatin_sample, expression_sample,
			chromatin_key=chromatin_key, xlim=xlim, ylim=ylim, lw=lw, 
			plot_arrows=plot_arrows, color_arrows=color_arrows)
	
	def plot_all_metrics_all_replicates_gene(self, gene_or_orf_name, figsize=(4, 4),
											save_plots=False, title=None):
		"""
		Create 9-panel plot showing all chromatin metrics vs expression for all data sources.
		
		Layout:
		Row 0: Nucleosome Occupancy    [Raw Rep 1] [Raw Rep 2] [Deconvolved]
		Row 1: Promoter Occupancy      [Raw Rep 1] [Raw Rep 2] [Deconvolved]  
		Row 2: Nucleosome Entropy      [Raw Rep 1] [Raw Rep 2] [Deconvolved]
		
		Parameters
		----------
		gene_or_orf_name : str
			Gene name or ORF name to plot
		figsize : tuple, optional
			Figure size
		save_plots : bool, optional
			Whether to save the plot
		"""
		from src.sgd import get_gene_title_name
		from src.plot_helpers import create_nine_subplot_layout
		
		# Create the layout
		fig, axes = create_nine_subplot_layout(figsize=figsize)
		
		# Define metrics and their plotting parameters
		metrics_config = {
			'nucleosome_occupancy': {
				'title': 'Nucleosome Occupancy',
				'xlim': (0, 5),
				'ylim': (0, 15)
			},
			'promoter_occupancy': {
				'title': 'Promoter Occupancy', 
				'xlim': (0, 3),
				'ylim': (0, 15)
			},
			'nucleosome_entropy': {
				'title': 'Nucleosome Entropy',
				'xlim': (0, 3),
				'ylim': (0, 15)
			}
		}
		
		metrics = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		data_sources = ['raw_rep1', 'raw_rep2', 'deconvolved']
		column_titles = ['Raw 1', 'Raw 2', 'Deconvolved']
		
		# Plot each metric-source combination
		for row_idx, metric in enumerate(metrics):
			metric_config = metrics_config[metric]
			
			for col_idx, (data_source, col_title) in enumerate(zip(data_sources, column_titles)):
				ax = axes[row_idx][col_idx]
				plt.sca(ax)  # Set current axis
				
				# Plot based on data source
				if data_source == 'raw_rep1':
					self.plot_orf_phase_state_raw(
						gene_or_orf_name, 
						chromatin_key=metric, 
						replicate=1, 
						xlim=metric_config['xlim'], 
						ylim=metric_config['ylim']
					)
				elif data_source == 'raw_rep2':
					self.plot_orf_phase_state_raw(
						gene_or_orf_name, 
						chromatin_key=metric, 
						replicate=2, 
						xlim=metric_config['xlim'], 
						ylim=metric_config['ylim']
					)
				else:  # deconvolved
					self.plot_orf_phase_state_deconvolved(
						gene_or_orf_name, 
						chromatin_key=metric, 
						xlim=metric_config['xlim'], 
						ylim=metric_config['ylim'],
						lw=4,
					)
				
				# Set titles for top row
				if row_idx == 0:
					ax.set_title(col_title, fontsize=12)
				else:
					ax.set_title('')  # Clear default title from individual plot functions

				ax.set_xticks([])
				ax.set_yticks([])
				
				# Set y-axis labels only for leftmost column
				if col_idx > 0:
					ax.set_ylabel('')
				else:
					ylabel = metric.replace('_', '\n')
					ylabel = ylabel[0].upper() + ylabel[1:]
					plt.ylabel(f"{ylabel}", rotation=0, ha='right', va='center',
						labelpad=20)

					# Position based on xlims
					xlims = metric_config['xlim']
					xlims_diff = xlims[1]-xlims[0]
					x_pos = xlims[0]-xlims_diff*0.125

					ax.text(x_pos, 0.12, 'Expression', rotation=90, ha='left', 
						va='bottom', fontsize=8)  # Bottom row
				
				# Set x-axis labels only for bottom row
				if row_idx < 2:
					ax.set_xlabel('')
				else:

					if col_idx == 0 or col_idx == 2:

						x_pos = 0

						ax.text(x_pos, -2.5, 'Chromatin value', ha='left', 
							va='bottom', fontsize=8)  # Bottom row
						ax.set_xlabel('')
					else:
						ax.set_xlabel('')

		# Add overall title with gene name
		if title is None:
			title = get_gene_title_name(gene_or_orf_name)

		fig.suptitle(f'{title}',
					fontweight='demi', fontsize=14)
		
		return fig
	
	def plot_single_metric_three_replicates(self, gene_or_orf_name, chromatin_key, 
										   figsize=(8, 2.5), xlim=(-0.5, 8), ylim=(-0.5, 12),
										   wspace_left=0, wspace_right=0.1, 
										   save_plots=False):
		"""
		Create 3-panel plot showing raw replicate 1, raw replicate 2, and deconvolved data
		for a single chromatin metric and single gene.
		"""
		from src.plot_helpers import create_three_subplot_layout
		from src.sgd import get_gene_title_name
		
		# Create the three subplot layout
		fig, ax1, ax2, ax3 = create_three_subplot_layout(
			figsize=figsize, 
			wspace_left=wspace_left, 
			wspace_right=wspace_right,
			top_padding=0.3
		)
		
		# Adjust xlim for specific metrics if needed (following the pattern from original code)
		if chromatin_key == 'promoter_occupancy':
			xlim = (-0.5, 4)
		elif chromatin_key == 'nucleosome_occupancy':
			xlim = (-0.5, 8)
		elif chromatin_key == 'nucleosome_entropy':
			xlim = (0, 4)
		
		# Plot raw replicate 1 data
		plt.sca(ax1)  # Set current axis
		self.plot_orf_phase_state_raw(gene_or_orf_name, chromatin_key=chromatin_key, 
									  replicate=1, xlim=xlim, ylim=ylim)
		ax1.set_title(f'Raw Replicate 1')
		
		# Plot raw replicate 2 data
		plt.sca(ax2)  # Set current axis
		self.plot_orf_phase_state_raw(gene_or_orf_name, chromatin_key=chromatin_key, 
									  replicate=2, xlim=xlim, ylim=ylim)
		ax2.set_title(f'Raw Replicate 2')
		ax2.set_ylabel(None)
		ax2.set_yticks([])
		
		# Plot deconvolved data
		plt.sca(ax3)  # Set current axis
		self.plot_orf_phase_state_deconvolved(gene_or_orf_name, chromatin_key=chromatin_key, 
											 xlim=xlim, ylim=ylim)
		ax3.set_title(f'Deconvolved')
		
		# Add overall title with gene name
		gene_title_name = get_gene_title_name(gene_or_orf_name)

		chromatin_title = chromatin_key.replace('_', ' ')
		chromatin_title = chromatin_title[:1].upper() + chromatin_title[1:]
		fig.suptitle(f'{gene_title_name} - {chromatin_title}', fontweight='demi', fontsize=12)
		
		return fig

	def create_gene_dataset_intersection_data(self, chromatin_key='nucleosome_occupancy'):
		"""
		Create and store the gene intersection dataset with groupings.
		Stores the result in self.gene_intersection_data[chromatin_key].
		"""
		
		# Extract gene sets from the integration object
		raw_genes1 = set(self.coordinated_genes['raw_rep1'][chromatin_key]['genes'])
		raw_genes2 = set(self.coordinated_genes['raw_rep2'][chromatin_key]['genes'])
		deconv_genes = set(self.coordinated_genes['deconvolved'][chromatin_key]['genes'])
		
		# Create union of raw genes
		raw_genes = raw_genes1.union(raw_genes2)
		all_genes = sorted(list(raw_genes.union(deconv_genes)))
		
		# Create binary matrix
		data = []
		for gene in all_genes:
			row = {
				'raw_rep1': 1 if gene in raw_genes1 else 0,
				'raw_rep2': 1 if gene in raw_genes2 else 0,
				'deconvolved': 1 if gene in deconv_genes else 0
			}
			data.append(row)
		
		df_matrix = pd.DataFrame(data, index=all_genes)
		
		# Classify genes into biological groups
		def classify_gene(row):
			in_raw1 = row['raw_rep1'] == 1
			in_raw2 = row['raw_rep2'] == 1
			in_deconv = row['deconvolved'] == 1
			in_any_raw = in_raw1 or in_raw2
			
			if in_raw1 and in_raw2 and in_deconv:
				return 'all_three'
			elif in_any_raw and in_deconv:
				return 'raw_and_deconv'
			elif in_any_raw and not in_deconv:
				return 'raw_only'
			elif not in_any_raw and in_deconv:
				return 'deconv_only'
			else:
				return 'neither'
		
		df_matrix['group'] = df_matrix.apply(classify_gene, axis=1)
		
		# Add explicit sorting columns
		df_matrix['sort_group_priority'] = df_matrix['group'].map({
			'all_three': 0,
			'raw_and_deconv': 1,
			'raw_only': 2,
			'deconv_only': 3,
			'neither': 4
		})

		# Secondary sort: raw replicate pattern (both > raw1_only > raw2_only)
		def get_raw_pattern_priority(row):
			if row['raw_rep1'] == 1 and row['raw_rep2'] == 1:
				return 0  # Both replicates - highest priority
			elif row['raw_rep1'] == 1 and row['raw_rep2'] == 0:
				return 1  # Raw 1 only - second priority
			elif row['raw_rep1'] == 0 and row['raw_rep2'] == 1:
				return 2  # Raw 2 only - third priority
			else:
				return 3  # Neither (shouldn't happen in your data)

		df_matrix['sort_raw_pattern'] = df_matrix.apply(get_raw_pattern_priority, axis=1)

		# Tertiary sort: gene name (alphabetical)
		df_matrix['sort_gene_name'] = df_matrix.index

		# Sort using pandas multi-column sorting
		df_matrix = df_matrix.sort_values([
			'sort_group_priority',           # Primary: group
			'sort_raw_pattern',              # Secondary: raw pattern (both > raw1 > raw2)
			'sort_gene_name'                 # Tertiary: gene name (alphabetical)
		], ascending=[True, True, True])     # All ascending

		# Remove the temporary sorting columns
		df_matrix = df_matrix.drop(['sort_group_priority', 'sort_raw_pattern', 'sort_gene_name'], axis=1)
		
		# Store the data in member variable
		if not hasattr(self, 'gene_intersection_data'):
			self.gene_intersection_data = {}
		
		self.gene_intersection_data[chromatin_key] = df_matrix
		
		return df_matrix


	def plot_gene_dataset_intersection_heatmap(self, chromatin_key='nucleosome_occupancy', figsize=(4, 5)):
		"""
		Create a heatmap showing which genes are present in raw vs deconvolved datasets.
		Uses pre-computed data from self.gene_intersection_data[chromatin_key].
		"""
		
		# Check if data exists
		if not hasattr(self, 'gene_intersection_data') or chromatin_key not in self.gene_intersection_data:
			raise ValueError(f"Gene intersection data for '{chromatin_key}' not found. "
							f"Please run create_gene_dataset_intersection_data() first.")
		
		df_matrix = self.gene_intersection_data[chromatin_key].copy()
		
		# Find group boundaries and create labels
		group_boundaries = []
		group_labels = []
		group_counts = {}
		current_group = None
		
		# Count genes in each group and find boundaries
		for i, (gene, row) in enumerate(df_matrix.iterrows()):
			group = row['group']
			group_counts[group] = group_counts.get(group, 0) + 1
			
			if current_group != group:
				if current_group is not None:
					group_boundaries.append(i - 0.5)
				current_group = group
		
		# Add group labels
		for group in ['all_three', 'raw_and_deconv', 'raw_only', 'deconv_only', 'neither']:
			if group in group_counts:
				if group == 'all_three':
					label = f"Full agreement (n={group_counts[group]})"
				elif group == 'raw_and_deconv':
					label = f"1 Raw+Deconvolved (n={group_counts[group]})"
				elif group == 'raw_only':
					label = f"Raw only (n={group_counts[group]})"
				elif group == 'deconv_only':
					label = f"Deconvolved only (n={group_counts[group]})"
				else:
					label = f"Other (n={group_counts[group]})"
				group_labels.append(label)
		
		# Create plotting dataframe (exclude group column)
		df_plot = df_matrix.drop('group', axis=1)
		
		# Create the heatmap
		fig, ax = plt.subplots(figsize=figsize)

		from pipeline.chromatin_metrics_processor import plot_formatting_map

		chromatin_cmap = plot_formatting_map[chromatin_key]['cmap']

		# Create column-specific colormaps
		colors_col1 = ['white', '#ddd']  # raw_rep1
		colors_col2 = ['white', '#ddd']  # raw_rep2  
		colors_col3 = ['white', plt.get_cmap(chromatin_cmap)(0.55)]  # deconvolved
		
		# Plot each column separately with different colors
		for col_idx, (col_name, colors) in enumerate(zip(df_plot.columns, 
				[colors_col1, colors_col2, colors_col3])):
			cmap = ListedColormap(colors)
			col_data = df_plot[col_name].values.reshape(-1, 1)
			im = ax.imshow(col_data, cmap=cmap, aspect='auto', interpolation='nearest',
						  extent=[col_idx-0.5, col_idx+0.5, len(df_plot)-0.5, -0.5])
		
		# Add red horizontal lines to separate groups
		for boundary in group_boundaries:
			ax.axhline(y=boundary, color='red', linewidth=0.75)
		
		# Add column boundaries
		ax.axvline(x=1.5, color='black', linewidth=0.75)
		ax.axvline(x=0.5, color='black', linewidth=0.5)
		
		# Customize the plot
		chromatin_title = chromatin_key.replace('_', ' ')
		chromatin_title = chromatin_title[:1].upper() + chromatin_title[1:]

		ax.set_title(f'{chromatin_title}, n={len(df_plot)}', 
					fontsize=14, fontweight='demi', pad=10)
		
		# Set x-axis labels
		ax.set_xticks([0, 1, 2])
		ax.set_xticklabels(['Raw 1', 'Raw 2', 'Deconv'], fontsize=10)
		ax.set_xlim((-0.5, 2.5))

		# Add group labels on the right
		ax2 = ax.twinx()
		ax2.set_ylim(ax.get_ylim())
		
		# Calculate midpoint positions for group labels
		label_positions = []
		start_idx = 0
		current_group = None
		
		for i, (gene, row) in enumerate(df_matrix.iterrows()):
			group = row['group']
			if current_group != group:
				if current_group is not None:
					# Add label at midpoint of previous group
					mid_pos = (start_idx + i - 1) / 2
					label_positions.append(mid_pos)
					start_idx = i
				current_group = group
		
		# Add final group
		if len(df_matrix) > 0:
			mid_pos = (start_idx + len(df_matrix) - 1) / 2
			label_positions.append(mid_pos)
		
		# Set group labels
		ax2.set_yticks(label_positions)
		ax2.set_yticklabels(group_labels, fontsize=8)
		ax2.tick_params(axis='y', which='major', pad=4, length=0)
		
		# Remove y-axis ticks from main plot
		ax.set_yticks([])
			
		plt.tight_layout()

		return fig


	def create_gene_dataset_intersection_heatmap(self, chromatin_key='nucleosome_occupancy', 
											   figsize=(4, 5)):
		"""
		Convenience function that creates the data and plots the heatmap in one call.
		"""
		self.create_gene_dataset_intersection_data(chromatin_key)
		return self.plot_gene_dataset_intersection_heatmap(chromatin_key, figsize)


	def compute_and_assign_linkages_data(self):
		"""Compute linkages data set for all chromatin keys"""

		self.trajectory_area_linkages = {}
		for chromatin_key in ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']:
			self.trajectory_area_linkages[chromatin_key] = self.compute_linkage_measures(chromatin_key)

	def compute_linkage_measures(self, chromatin_key):
		"""Create a dataframe of PTRs and trajectory area for a given chromatin
		key and link with expression.
		"""
		from pipeline.integration_cycling_calculations import calculate_linkage_values

		# Analyze shapes of promoter occupancy vs transcription
		expression_data = self.expression_processor.expression_data
		expression_data.columns = expression_data.columns.astype(int)

		chromatin_data = self.chromatin_processor\
			.normalized_deconvolved_metrics[chromatin_key]
		chromatin_data.columns = chromatin_data.columns.astype(int)
		chromatin_data = chromatin_data.dropna()

		t_indices = self.config.t_indices()
		b_indices = self.config.b_indices()
		common_index = expression_data[[]].join(chromatin_data[[]], how='inner').index

		# Compute the trajectory area for the mother and daughter branches
		t_expression = expression_data[t_indices].loc[common_index].values
		t_chromatin = chromatin_data[t_indices].loc[common_index].values

		b_expression = expression_data[b_indices].loc[common_index].values
		b_chromatin = chromatin_data[b_indices].loc[common_index].values
		tb_expression = t_expression+b_expression
		tb_chromatin = t_chromatin+b_chromatin

		res = calculate_linkage_values(tb_expression, tb_chromatin, index=common_index)

		expression_ptrs = self.expression_processor.get_genic_ptrs()
		chromatin_ptrs = self.chromatin_processor\
			.normalized_ptr_deconvolved[chromatin_key]

		ptrs_joined = expression_ptrs[['ptr']].join(chromatin_ptrs, how='inner')
		ptrs_joined.columns = ['expression_ptr', 'chromatin_ptr']
		ptrs_joined = ptrs_joined.join(res)
		ptrs_joined = ptrs_joined.sort_values('normalized_trajectory_area')

		return ptrs_joined

	def plot_chromatin_traj_area_genes(self, chromatin_key):
		
		from pipeline.chromatin_metrics_processor import plot_formatting_map
		format_map = plot_formatting_map[chromatin_key]
		cmap = format_map['cmap']
		name = format_map['name']
		
		plot_res_data = self.trajectory_area_linkages[chromatin_key]
		selected_orf_names = self.coordinated_genes['deconvolved'][chromatin_key]['genes']
		selected_data = plot_res_data.loc[selected_orf_names].sort_values('normalized_trajectory_area')

		gene_names = [get_gene_title_name(orf_name, include_system=False) \
						  for orf_name, row in selected_data.iterrows()]

		plt.scatter(selected_data.normalized_trajectory_area, gene_names, 
					s=2, marker='D', color=plt.get_cmap(cmap)(0.7))
		plt.title(f"{name}, n={len(selected_data)}")
		ys = np.arange(len(gene_names))
		plt.yticks(ys, gene_names, fontsize=6)
		plt.ylim(-0.5, len(gene_names)-0.5)
		plt.xlabel("Trajectory area")
		
		return selected_data

	def plot_all_trajectory_area_cell_cycle_genes(self):
		plt.figure(figsize=(4, 16))
		plt.subplot(3, 1, 1)
		top_prom_occ = self.plot_chromatin_traj_area_genes('promoter_occupancy')
		plt.xlabel('')

		plt.subplot(3, 1, 2)
		self.plot_chromatin_traj_area_genes('nucleosome_entropy')
		plt.xlabel('')

		plt.subplot(3, 1, 3)
		self.plot_chromatin_traj_area_genes('nucleosome_occupancy')

		plt.suptitle("Cell cycling chromatin and\ntranscription genes, Trajectory area", 
			fontweight='demi', fontsize=18, y=0.95)
		plt.subplots_adjust(hspace=0.25)


	# Plot the highest to lowest trajectory area genes, in the coordinated genes set
	# as a small grid of phase-state plots

	def plot_deconvolved_examples_grid(self, orf_names, chromatin_key):
		
		fig = plt.figure(figsize=(11, 2.75))
		nrows, ncols = 2, 10
		n_tot = nrows*ncols
		
		ylim = trajectory_lims_mapping['expression']
		xlim = trajectory_lims_mapping[chromatin_key]
		
		orf_names = list(orf_names[:ncols]) + list(orf_names[-ncols:])

		from src.sgd import get_gene_title_name
		
		for i, orf_name in enumerate(orf_names):

			plt.subplot(nrows, ncols, i+1)
			self.plot_orf_phase_state_deconvolved(orf_name, chromatin_key,
																plot_arrows=False,
																	xlim=xlim, ylim=ylim,
																	lw=2)
			plt.title('')
			plt.xlabel('')
			plt.ylabel('')
			plt.xticks([])
			plt.yticks([])
			
			gene_title = get_gene_title_name(orf_name, include_system=False)
			plt.text(xlim[0]+(xlim[1]-xlim[0])*.05, 
					 ylim[1]-(ylim[1]-ylim[0])*0.05, gene_title, va='top',
					 fontsize=10)

			# Label the axes
			if i == 0:

				# Group labeling
				plt.ylabel(f"Bottom {ncols}", rotation=0, ha='right', labelpad=22, fontsize=12)
			elif i == ncols:

				# Group labeling
				plt.ylabel(f"Top {ncols}", rotation=0, ha='right', labelpad=22, fontsize=12)

				# Values labeling
				xlabel = chromatin_key.replace('_', ' ')
				xlabel = xlabel[0].upper() + xlabel[1:]
				plt.xlabel(xlabel, fontsize=10, labelpad=7, ha='left', x=0)

		plt.subplots_adjust(wspace=0, hspace=0.3, top=0.81, left=0.15)  # More left margin

		fig.text(0.131, 0.12, 'Expression', rotation=90, ha='left', 
			va='bottom', fontsize=10)  # Bottom row

	def plot_chromatin_example_trajectories(self, chromatin_key):
		
		from pipeline.chromatin_metrics_processor import plot_formatting_map
		name = plot_formatting_map[chromatin_key]['name']
		
		plot_res_data = self.trajectory_area_linkages[chromatin_key]
		selected_orf_names = self.coordinated_genes['deconvolved'][chromatin_key]['genes']
		selected_data = plot_res_data.loc[selected_orf_names].sort_values('normalized_trajectory_area',
																		 ascending=True)
		print("Number of genes", len(selected_data))
		
		self.plot_deconvolved_examples_grid(selected_data.index.values, chromatin_key)
		plt.suptitle(f"{name}, by normalized trajectory area", fontweight='demi', fontsize=16)

	def _collect_trajectory_lims(self, orf_or_gene_name, chromatin_key):
		"""
		Collect all chromatin and expression values that will be plotted for a gene.
		Used for limits calculation
		"""
		import numpy as np
		from src.sgd import get_gene_name_orf_name
		
		# Get chromatin and expression data sources
		chromatin_metrics_data = self.chromatin_processor.normalized_deconvolved_metrics
		deconvolved_chromatin_data = chromatin_metrics_data[chromatin_key]
		deconvolved_transcription_data = self.expression_processor.expression_data
		
		# Resolve ORF name
		if orf_or_gene_name in chromatin_metrics_data[chromatin_key].index:
			orf_name = orf_or_gene_name
		else:
			orf_name, gene_name = get_gene_name_orf_name(orf_or_gene_name)
		
		# Extract time course data for this gene
		chromatin_sample = deconvolved_chromatin_data.loc[orf_name]
		expression_sample = deconvolved_transcription_data.loc[orf_name]
		
		# Initialize lists to collect all values
		all_chromatin_values = []
		all_expression_values = []
		
		# 1. Collect connecting line values (averaged t/b branches)
		t_indices = self.config.get_Hpositions_for_branch('t')
		b_indices = self.config.get_Hpositions_for_branch('b')
		expression_values = (expression_sample[t_indices].values + 
							 expression_sample[b_indices].values) / 2
		chromatin_values = (chromatin_sample[t_indices].values + 
							chromatin_sample[b_indices].values) / 2
		# Concatenate first point to end (closing the loop)
		expression_values = np.concatenate([expression_values, expression_values[0:1]])
		chromatin_values = np.concatenate([chromatin_values, chromatin_values[0:1]])
		
		all_chromatin_values.append(chromatin_values)
		all_expression_values.append(expression_values)
		
		# 2. Collect phase-specific values
		phases = ['G2M', 'S', 'meanG1']
		
		for phase in phases:
			if phase == 'meanG1':
				# Average CG1 and DG1 branches
				t_indices = self.config.get_Hpositions_for_phase('CG1')
				b_indices = self.config.get_Hpositions_for_phase('DG1')
				
				expression_values = (expression_sample[t_indices].values + 
									 expression_sample[b_indices].values) / 2
				chromatin_values = (chromatin_sample[t_indices].values + 
									chromatin_sample[b_indices].values) / 2
			else:
				# Direct indices for G2M and S phases
				indices = self.config.get_Hpositions_for_phase(phase)
				expression_values = expression_sample[indices].values
				chromatin_values = chromatin_sample[indices].values
			
			all_chromatin_values.append(chromatin_values)
			all_expression_values.append(expression_values)
		
		# 3. Flatten all collected values into single arrays
		chromatin_values_flat = np.concatenate(all_chromatin_values)
		expression_values_flat = np.concatenate(all_expression_values)
		
		return chromatin_values_flat, expression_values_flat

	def compute_trajectory_limits(self, gene_list, metrics=None, lim_padding=0.25):
		"""
		Compute x and y axis limits for trajectory plots based on a list of genes.
		
		This method collects all data points that would be plotted for the specified genes
		and computes appropriate axis limits with padding. Useful for ensuring multiple
		plots share the same axis limits.
		"""
		import numpy as np
		
		# 1. Set default metrics if not provided
		if metrics is None:
			metrics = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		
		# 2. Initialize data collectors
		all_chromatin_data = {metric: [] for metric in metrics}
		all_expression_data = []
		
		# 3. Collect all plot values
		for gene in gene_list:
			for metric in metrics:
				chrom_vals, expr_vals = self._collect_trajectory_lims(gene, metric)
				all_chromatin_data[metric].extend(chrom_vals)
				all_expression_data.extend(expr_vals)

		# 4. Define limit calculation helper
		def _create_lims(data, padding):
			"""
			Compute axis limits from data with padding.
				
			Returns
			-------
			tuple
				(lower_limit, upper_limit)
			"""
			min_max = np.quantile(data, q=[0, 1])
			value_range = min_max[1] - min_max[0]
			ret = (min_max[0] - padding * value_range,
					min_max[1] + padding * value_range)
			return ret
		
		# 5. Compute expression limits (y-axis)
		from src.math_utils import create_data_lims
		ylim = create_data_lims(all_expression_data, lim_padding)
		
		# 6. Compute chromatin limits for each metric (x-axis)
		xlim_dict = {}
		for metric in metrics:
			xlim_dict[metric] = _create_lims(all_chromatin_data[metric], lim_padding)

		# 7. Return the computed limits
		return xlim_dict, ylim

	def plot_gene_group_trajectories_all_metrics(self, gene_list, 
			figsize_per_row=(6, 1.5), title='', auto_lims=False,
			override_xlim=None, override_ylim=None):
		"""
		Plot trajectories for a set of genes across all three chromatin metrics.
		
		Creates an N×3 grid where each row is a gene and columns are:
		[Promoter Occupancy, Nucleosome Entropy, Nucleosome Occupancy]
		
		Parameters
		----------
		gene_list : list
			List of gene names or ORF names to plot in desired order
		figsize_per_row : tuple, optional
			Figure size per row (width, height), total figure scales with number of genes
			
		Returns
		-------
		matplotlib.figure.Figure
			The created figure object
		"""
		
		# Define the metrics and their axis limits
		metrics = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		metric_titles = ['Promoter Occupancy', 'Nucleosome Entropy', 'Nucleosome Occupancy']
		
		# Calculate total figure size
		n_genes = len(gene_list)

		# With this - add fixed space for title:
		title_space = 1.1  # Fixed inches for title area
		additional_title_spacing = n_genes/10 * 0.3 # Scale the title space by the number of rows
		title_space = title_space + additional_title_spacing

		total_figsize = (figsize_per_row[0], figsize_per_row[1] * n_genes + title_space)
		
		# Create the subplot grid
		fig, axes = plt.subplots(n_genes, 3, figsize=total_figsize)
		
		# Handle case of single gene (axes won't be 2D)
		if n_genes == 1:
			axes = axes.reshape(1, -1)

		# Compute the limits based on each chromatin measures
		# min and max ranges
		if auto_lims:
			auto_xlims, auto_ylims = \
				self.compute_trajectory_limits(gene_list, metrics=None)

		# Plot each gene-metric combination
		for row_idx, gene in enumerate(gene_list):
			for col_idx, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
				ax = axes[row_idx, col_idx]
				plt.sca(ax)  # Set current axis
				
				if auto_lims:

					xlim = auto_xlims[metric]
					ylim = auto_ylims

				else:
					xlim = override_xlim[metric]
					ylim = override_ylim

				# Plot the trajectory
				self.plot_orf_phase_state_deconvolved(
					gene, 
					chromatin_key=metric, 
					xlim=xlim, 
					ylim=ylim,
					plot_arrows=False
				)
				
				# Clear the default title from individual plot function
				ax.set_title('')
				
				# Set column headers only for top row
				if row_idx == 0:
					ax.set_title(metric_title.replace(' ', '\n'), fontsize=14)

				ax.set_ylabel('')
				
				# Set gene names only for leftmost column
				if col_idx == 0:
					from src.sgd import get_gene_title_name
					gene_title_name = get_gene_title_name(gene, include_system=False)
					ax.set_ylabel(gene_title_name, rotation=0, ha='right', fontsize=24,
						labelpad=7)

				elif col_idx == 2:
					axis_label_ax = ax.twinx()
					axis_label_ax.set_yticks([])
					axis_label_ax.set_ylabel('Expression', fontsize=14, rotation=270,
						labelpad=12, ha='center')

				ax.set_yticks([])
				ax.set_xticks([])
				ax.set_xlabel('')

		# Then at the end, use subplots_adjust with calculated top margin:
		plot_area_height = figsize_per_row[1] * n_genes
		top_margin = plot_area_height / total_figsize[1]  # This will be consistent

		plt.suptitle(title, fontweight='demi', fontsize=26)
		plt.subplots_adjust(wspace=0, hspace=0, top=top_margin, bottom=0)
		
		return fig


	# Here I begin an updated trajectory analysis.
	#
	# Design:
	# 1. Selection of dual cyclic genes (cycling in chromatin and expression)
	# 2. Computation of trajectory area and slope for these genes
	# 3. Two tiered analysis: 
	#    i. Slope of small trajectory area genes
	#    ii. Directionality of large trajectory area genes
	# 
	def select_data_for_trajectory_analysis(self):

		# We'll hold state and process these metrics one at a time
		# todo: for now...
		chromatin_key = 'promoter_occupancy'

		ptr_q = 0.9

		joined_data = self.correlation_results['deconvolved'][chromatin_key]['joined_data'].copy()
		chrom_threshold = np.quantile(joined_data.chromatin_ptr, q=ptr_q)
		tx_threshold = np.quantile(joined_data.expression_ptr, q=ptr_q)

		joined_data['is_chrom_cycling'] = joined_data.chromatin_ptr > chrom_threshold
		joined_data['is_tx_cycling'] = joined_data.expression_ptr > tx_threshold

		both_cycling = joined_data[joined_data.is_chrom_cycling & joined_data.is_tx_cycling]
		print(f"Number of genes cycling in {chromatin_key} and transcription", len(both_cycling))

		self.current_joined_chromatin_data = joined_data
		self.current_chromatin_key = chromatin_key
		self.current_chrom_threshold = chrom_threshold
		self.expression_threshold = tx_threshold

	def _select_chrom_tx_data(self, orf_name, chrom_key, config=None):
		"""Helper function to select the chromatin and expression data
		for an orf, averaging the mother/daughter branches"""

		if config is None:
			from src.config import load_default_chrom_configs
			config, _ = load_default_chrom_configs()    

		chrom_data = self.chromatin_processor.deconvolved_chromatin_metrics[\
			chrom_key]
		expression_data = self.expression_processor.expression_data

		selected_gene_chromatin_data = chrom_data.loc[orf_name]
		selecetd_gene_expression_data = expression_data.loc[orf_name]

		def _get_mean_tb_data(data):
			data = data.copy()
			data.index = data.index.astype(int)
			return (data[config.t_indices()].values + data[config.b_indices()].values)/2

		chromatin_values = _get_mean_tb_data(selected_gene_chromatin_data)
		expression_values = _get_mean_tb_data(selecetd_gene_expression_data)
		return chromatin_values, expression_values

	def compute_trajectory_metrics_all_genes(self):

		from src.config import load_default_chrom_configs
		config1, _ = load_default_chrom_configs()    

		from src.math_utils import ordinary_least_squares
		from src.math_utils import compute_signed_area
			
		slope_traj_data = self.trajectory_area_linkages[self.current_chromatin_key]\
			[['normalized_trajectory_area']].loc[self.current_joined_chromatin_data.index]

		for i, (orf_name, row) in enumerate(slope_traj_data.iterrows()):
			chromatin_values, expression_values = self._select_chrom_tx_data(orf_name, 
				self.current_chromatin_key, config=config1)
			ols_slope, ols_intercept = ordinary_least_squares(chromatin_values, expression_values)    
			signed_area = compute_signed_area(chromatin_values, expression_values)
			
			slope_traj_data.loc[orf_name, 'ols_slope'] = ols_slope
			slope_traj_data.loc[orf_name, 'ols_intercept'] = ols_intercept
			slope_traj_data.loc[orf_name, 'signed_area'] = signed_area
			slope_traj_data.loc[orf_name, 'is_ccw'] = signed_area > 0

			if i % 1000 == 0:
				print(f"{i}/{len(slope_traj_data)}")

		# Set threshold for promoter occupancy manually, based on visual
		# inspection for separating genes
		self.current_traj_threshold = 0.07
		slope_traj_data['is_large_area'] = \
			slope_traj_data.normalized_trajectory_area >= self.current_traj_threshold

		self.current_slope_trajectory_data = slope_traj_data

	def plot_slope_trajectory_data(self):

		slope_traj_data = self.current_slope_trajectory_data
		joined_data = self.current_joined_chromatin_data
		both_cycling = joined_data[
			joined_data.is_chrom_cycling &
			joined_data.is_tx_cycling].index

		plt.figure(figsize=(8.5, 3.5))

		plt.subplot(1, 2, 1)
		ax = plt.gca()

		from src.DensityScatterPlotter import DensityScatterPlotter
		cmap = plt.cm.Oranges
		density_scatter_pltr = DensityScatterPlotter()
		density_scatter_pltr.bw = [0.025, 0.05]
		density_scatter_pltr.cmap = cmap
		density_scatter_pltr.alpha = 1.
		density_scatter_pltr.logz = True
		density_scatter_pltr.set_data(joined_data.chromatin_ptr,
			joined_data.expression_ptr)
		density_scatter_pltr.plot_ax(ax, plot_colorbar=False, vmin=0, 
			vmax=10)

		plt.title(f"Promoter occupancy cyclicity\n"
			f"vs transcription cyclicity, n={len(joined_data)}", fontweight='demi')
		plt.xlabel("Promoter occupancy PTR")
		plt.ylabel("Gene expression PTR")

		plt.axvline(self.current_chrom_threshold, c='black', lw=0.75, ls='dotted')
		plt.axhline(self.expression_threshold, c='black', lw=0.75, ls='dotted')

		ylim = 0.9, 6
		xlim = 0.95, 2.5
		plt.xlim(*xlim)
		plt.ylim(*ylim)

		# Place the counts as text annotations
		num_both = len(both_cycling)
		num_chrom_only = len(joined_data[joined_data.is_chrom_cycling & 
			~joined_data.is_tx_cycling])
		num_tx_only = len(joined_data[~joined_data.is_chrom_cycling & 
			joined_data.is_tx_cycling])
		neither = len(joined_data[(
			~joined_data.is_chrom_cycling & 
			~joined_data.is_tx_cycling)])

		y_text_positions = 1.25, ylim[1]*0.75
		x_text_positions = 1.15, xlim[1]*0.75

		plt.text(x_text_positions[1], y_text_positions[1], f"Co-cycling\n{num_both}",
			ha='center', va='center')
		plt.text(x_text_positions[1], y_text_positions[0], f"Promoter only, {num_chrom_only}",
			ha='center', va='center')
		plt.text(x_text_positions[0], y_text_positions[1], f"Expression only\n{num_tx_only}",
			rotation=90, ha='center', va='center')
		plt.text(x_text_positions[0], y_text_positions[0], f"{neither}",
			ha='center', va='center')

		plt.subplot(1, 2, 2)
		plt.scatter(slope_traj_data.ols_slope, 
					slope_traj_data.normalized_trajectory_area, s=5, 
					color=plt.cm.Greys(0.2), alpha=0.1)

		both_cycling_traj_data = slope_traj_data.loc[both_cycling]

		large_area_genes = both_cycling_traj_data[
			both_cycling_traj_data.is_large_area]
		small_area_genes = both_cycling_traj_data[
			~both_cycling_traj_data.is_large_area]

		large_ccw = large_area_genes[~large_area_genes.is_ccw]
		large_cw = large_area_genes[large_area_genes.is_ccw]

		# Tier 1: Concurrent cyclers
		n = len(small_area_genes)
		plt.scatter(small_area_genes.ols_slope, 
					small_area_genes.normalized_trajectory_area, s=9, 
					color='black', marker='x', lw=0.75, label=f"Concurrent, n={n}")

		# Tier 2: Delayed cyclers
		n = len(large_ccw)
		plt.scatter(large_ccw.ols_slope, 
					large_ccw.normalized_trajectory_area, facecolor='none',
					s=13, edgecolor='red', marker='o', lw=0.75, label=f"Delayed activation (CCW), n={n}")

		n = len(large_cw)
		plt.scatter(large_cw.ols_slope, 
					large_cw.normalized_trajectory_area, s=13, 
					facecolor='none',
					edgecolor='blue', marker='o', lw=0.75, label=f"Delayed repression (CW), n={n}")

		# todo: hardcoded trajectory area threshold for promoter occupancy
		plt.axhline(self.current_traj_threshold, c='black', lw=0.75, alpha=0.5, ls='dotted')

		plt.xlim(-25, 25)
		plt.ylim(-0.01, 0.45)
		plt.axvline(0, c='black', lw=0.75, alpha=0.5, ls='solid', zorder=0)
		plt.title(f"Slope vs trajectory area,"
			f"\nn={len(both_cycling_traj_data)} co-cycling genes",
			fontweight='demi')
		plt.xlabel("Concurrent regulation (slope)")
		plt.ylabel("Temporal offset (trajectory area)")
		plt.legend()

		xticks = [-20, 0, 20]
		xtick_labels = ['Repression', '0', 'Activation']

		plt.xticks(xticks, xtick_labels)

		plt.subplots_adjust(wspace=0.35)

	def plot_orf_trajectory(self, sort_idx=None):

		joined_traj_data = self.current_joined_chromatin_data.join(
			self.current_slope_trajectory_data)[['is_chrom_cycling', 
				'is_tx_cycling', 'ols_slope', 'normalized_trajectory_area',
			'is_large_area']]

		both_cycling = joined_traj_data[joined_traj_data.is_chrom_cycling & joined_traj_data.is_tx_cycling]\
			.sort_values(['is_large_area', 'normalized_trajectory_area'])

		# Let's try modifying the area by normalizing it with a pseudocount
		dat = self.trajectory_area_linkages['promoter_occupancy'].loc[both_cycling.index]
		dat['normalized_trajectory_area'] = dat['trajectory_area']/\
			(dat['diameter_sq'])
		dat = dat.sort_values('normalized_trajectory_area')

		both_cycling = dat[['trajectory_area', 'diameter_sq', 'bounding_box_area']].join(
			both_cycling, how='left')

		chromatin_key = 'promoter_occupancy'

		# If predefined sorting order
		if sort_idx is not None:
			both_cycling = both_cycling.loc[sort_idx]

		plt.figure(figsize=(11, 12))
		for i, (orf_name, row) in enumerate(both_cycling.iterrows()):

			color_arrows = row.is_large_area

			plt.subplot(11, 8, i+1)

			# Normalize to mean center for plotting visilibity
			chromatin_data = self.chromatin_processor.deconvolved_chromatin_metrics[\
				'promoter_occupancy'].loc[orf_name]
			m, s = chromatin_data.mean(), chromatin_data.std()	
			chromatin_data = (chromatin_data-m)

			expression_data = self.expression_processor.expression_data.loc[orf_name]
			m, s = expression_data.mean(), expression_data.std()	
			expression_data = (expression_data-m)

			plot_orf_phase_state_deconvolved_values(self.config, 
				chromatin_data, expression_data,
				'promoter_occupancy',
				lw=2, xlim=(-1.5, 1.5), ylim=(-4, 4),#xlim=None, ylim=None,
				plot_arrows=True, color_arrows=color_arrows)

			from src.sgd import get_gene_name

			gene_name = get_gene_name(orf_name)
			if gene_name is None: gene_name = orf_name
			plt.text(-1.5, 3, f"{gene_name}")
			#\nN.area: {row.normalized_trajectory_area:.2f}\n"
			#	f"Area:{row.trajectory_area:.2f}\nDiam2:{row.diameter_sq:.2f}",
			#	ha='left', va='top')

			plt.xlabel('')
			plt.ylabel('')
			plt.title('')
			plt.xticks([])
			plt.yticks([])

		plt.subplots_adjust(wspace=0, hspace=0)
