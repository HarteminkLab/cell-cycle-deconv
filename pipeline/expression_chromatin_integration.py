import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import load_default_chrom_configs
from src.utils import print_fl, mkdir_safe
from src.figure_configs import save_figure_for_paper
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import ListedColormap

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
		
	def validate_data_availability(self):
		"""
		Validate that both processors have the required data loaded.
		
		Returns
		-------
		bool
			True if all required data is available
		"""
		required_chromatin_attrs = [
			'deconvolved_chromatin_metrics',
			'deconvolved_chromatin_ptrs'
		]
		required_expression_attrs = [
			'expression_data',
			'genic_ptrs'
		]
		
		missing_attrs = []
		
		for attr in required_chromatin_attrs:
			if not hasattr(self.chromatin_processor, attr) or \
			   getattr(self.chromatin_processor, attr) is None:
				missing_attrs.append(f"chromatin_processor.{attr}")
		
		for attr in required_expression_attrs:
			if not hasattr(self.expression_processor, attr) or \
			   getattr(self.expression_processor, attr) is None:
				missing_attrs.append(f"expression_processor.{attr}")
		
		if missing_attrs:
			raise ValueError(f"Missing required data: {missing_attrs}")
		
		return True
	
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
		self.validate_data_availability()
		
		# Get chromatin PTRs based on data source
		if chromatin_data_source == 'deconvolved':
			chromatin_ptrs = self.chromatin_processor.deconvolved_chromatin_ptrs
		elif chromatin_data_source == 'raw_rep1':
			chromatin_ptrs = self.chromatin_processor.raw_rep1_ptrs
		elif chromatin_data_source == 'raw_rep2':
			chromatin_ptrs = self.chromatin_processor.raw_rep2_ptrs
		else:
			raise ValueError(f"Unknown chromatin data source: {chromatin_data_source}")
		
		# Get expression PTRs (filtered to genic transcripts)
		expression_ptrs = self.expression_processor.genic_ptrs
		
		# Compute correlations for each chromatin metric
		correlation_results = {}
		
		for metric_name in ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']:
			metric_ptrs = chromatin_ptrs[metric_name]

			joined_ptrs = metric_ptrs.join(expression_ptrs, how='inner')
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
							  ptr_threshold=1.25,
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
		titles = ['Promoter Occupancy', 'Nucleosome Entropy', 'Nucleosome Occupancy']
		
		for i, (metric_name, title) in enumerate(zip(metric_names, titles)):
			if metric_name not in results:
				continue
				
			ax = axes[i]
			results_for_metric = results[metric_name]
			data = results_for_metric['joined_data']

			from src.DensityScatterPlotter import DensityScatterPlotter

			dsc_plotter = DensityScatterPlotter()
			dsc_plotter.plot_outline = True
			dsc_plotter.outline_color = '#eee'
			dsc_plotter.logz=True

			from pipeline.chromatin_metrics_processor import plot_formatting_map
			cmap = plot_formatting_map[metric_name]['cmap']
			chrom_bw = plot_formatting_map[metric_name]['bw']
			ptr_lims = plot_formatting_map[metric_name]['ptr_lims']

			# Plot the PTRs
			dsc_plotter.set_data(data.chromatin_ptr, data.expression_ptr)
			dsc_plotter.bw = chrom_bw, 0.25
			dsc_plotter.cmap = cmap
			dsc_plotter.s = 5
			dsc_plotter.plot_ax(ax)
			
			# Labels and title
			ax.set_xlabel(f'{title} PTR')
			ax.set_ylabel('Expression PTR')
			ax.set_title(f'{title}\nr={results_for_metric["pearson_r"]:.3f}, '
				f'n={results_for_metric["n_genes"]}')

			ax.plot([ptr_threshold, 10], [ptr_threshold, ptr_threshold], c='red', ls='dotted', lw=1)
			ax.plot([ptr_threshold, ptr_threshold], [ptr_threshold, 10], c='red', ls='dotted', lw=1)

			ax.set_xlim(*ptr_lims)
			ax.set_ylim(0.7, 6)
		
		plt.suptitle(f'Chromatin vs Expression PTR concordance ({chromatin_data_source})', 
					fontweight='demi', fontsize=14)
		plt.tight_layout()
		
		return fig
	
	def identify_coordinated_genes(self, chromatin_data_source='deconvolved',
								  ptr_threshold=1.1, correlation_threshold=0.3):
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
		
		results = self.correlation_results[chromatin_data_source]
		coordinated_genes = {}
		
		for metric_name, data in results.items():
			# Filter genes with high PTRs in both chromatin and expression
			high_chromatin_mask = data['joined_data'].chromatin_ptr > ptr_threshold
			high_expression_mask = data['joined_data'].expression_ptr > ptr_threshold
			high_both_mask = high_chromatin_mask & high_expression_mask
			
			coordinated_genes[metric_name] = {
				'genes': data['joined_data'].loc[high_both_mask].index,
				'masked_ptrs': data['joined_data'].loc[high_both_mask],
				'n_genes': np.sum(high_both_mask)
			}
			
			print_fl(f"{metric_name}: {np.sum(high_both_mask)} coordinated genes "
					f"(PTR > {ptr_threshold})")
		
		self.coordinated_genes[chromatin_data_source] = coordinated_genes
		return coordinated_genes

	def plot_orf_phase_state_raw(self, orf_or_gene_name, chromatin_key=None, 
		replicate=1, xlim=(-0.5, 8), ylim=(-0.5, 12)):
		# Get chromatin PTRs based on data source

		if replicate == 1:
			chromatin_metrics_data = self.chromatin_processor.raw_rep1_metrics[chromatin_key]
			raw_transcription_data = self.expression_processor.raw_rep1_expression_data
		elif replicate == 2:
			chromatin_metrics_data = self.chromatin_processor.raw_rep2_metrics[chromatin_key]
			raw_transcription_data = self.expression_processor.raw_rep2_expression_data
		else:
			raise ValueError()

		from src.sgd import get_gene_name_orf_name, get_gene_title_name

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

		plt.scatter(chromatin_sample, expression_sample, s=12, lw=2,
			edgecolor='#aaa', facecolor='none', zorder=2)
		plt.scatter(chromatin_sample, expression_sample, s=10, c=z, cmap=cmap, zorder=2)
		
		plt.xlabel(f"{chromatin_key}")
		plt.ylabel(f"Raw expression")
		plt.xlim(*xlim)
		plt.ylim(*ylim)
		plt.title(chromatin_key)
	
	def plot_orf_phase_state_deconvolved(self, orf_or_gene_name, chromatin_key=None, 
		xlim=(-0.5, 8), ylim=(-0.5, 12)):
		"""
		Plot chromatin vs expression data colored by cell cycle phase for a single gene.
		
		Parameters
		----------
		orf_or_gene_name : str
			Gene name or ORF name to plot
		chromatin_key : str, optional
			Which chromatin metric to plot ('promoter_occupancy', 'nucleosome_entropy', 
			'nucleosome_occupancy')
		xlim : tuple, optional
			X-axis limits
		ylim : tuple, optional
			Y-axis limits
		"""

		# Get chromatin PTRs based on data source
		chromatin_metrics_data = self.chromatin_processor.deconvolved_chromatin_metrics

		from src.sgd import get_gene_name_orf_name, get_gene_title_name
		
		# Get data
		deconvolved_chromatin_data = chromatin_metrics_data[chromatin_key]
		deconvolved_transcription_data = self.expression_processor.expression_data
		
		orf_name, gene_name = get_gene_name_orf_name(orf_or_gene_name)
		
		# Extract time course data
		chromatin_sample = deconvolved_chromatin_data.loc[orf_name]
		expression_sample = deconvolved_transcription_data.loc[orf_name]
		
		# Plot by cell cycle phase
		from src.plot_helpers import color_for_key
		phases = ['CG1', 'S', 'G2M']
		
		for phase in phases:
			indices = self.config.get_Hpositions_for_phase(phase)
			color = color_for_key(phase)

			plt.scatter(chromatin_sample[indices], expression_sample[indices],
					   s=1, color=color, label=phase)
		
		plt.legend()
		plt.xlabel(f"{chromatin_key}")
		plt.ylabel(f"Deconvolved expression")
		plt.xlim(*xlim)
		plt.ylim(*ylim)
		plt.title(chromatin_key)
	
	def plot_state_phase_all_metrics_gene(self, gene_or_orf_name, figsize=(9, 3),
			chromatin_data_source='deconvolved', save_plots=False):
		"""
		Create 3-panel plot showing all chromatin metrics vs expression for a single gene.
		
		Parameters
		----------
		gene_or_orf_name : str
			Gene name or ORF name to plot
		figsize : tuple, optional
			Figure size
		save_plots : bool, optional
			Whether to save the plot
		"""
		if chromatin_data_source == 'deconvolved':
			plot_metric_function = self.plot_orf_phase_state_deconvolved
			kwargs = {}
		elif chromatin_data_source == 'raw_rep1':
			plot_metric_function = self.plot_orf_phase_state_raw
			kwargs = {'replicate': 1}
		elif chromatin_data_source == 'raw_rep2':
			plot_metric_function = self.plot_orf_phase_state_raw
			kwargs = {'replicate': 2}
		else:
			raise ValueError()

		fig = plt.figure(figsize=figsize)
		
		plt.subplot(1, 3, 1)
		plot_metric_function(gene_or_orf_name, chromatin_key='nucleosome_occupancy', **kwargs)
		
		plt.subplot(1, 3, 2)
		plot_metric_function(gene_or_orf_name, chromatin_key='promoter_occupancy', 
								 xlim=(-0.5, 3), **kwargs)
		
		plt.subplot(1, 3, 3)
		plot_metric_function(gene_or_orf_name, chromatin_key='nucleosome_entropy', **kwargs,
			xlim=(1, 9))
		
		from src.sgd import get_gene_title_name
		gene_title_name = get_gene_title_name(gene_or_orf_name)
		
		plt.suptitle(gene_title_name, fontweight='demi', fontsize=12)
		plt.tight_layout()

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
			xlim = (-0.5, 3)
		elif chromatin_key == 'nucleosome_entropy':
			xlim = (1, 9)
		
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

		ax.set_title(f'{chromatin_title}', 
					fontsize=14, fontweight='demi', pad=20)
		
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
