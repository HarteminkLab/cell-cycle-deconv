import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.plot_helpers import create_proportional_subplots_3rows, create_proportional_subplots,\
	get_truncated_RdBu_r
from src.histones import HistoneModificationOrganizer, histones_ordering


class HistoneModificationGroupedPlotter:
	"""
	A class for plotting histone modification enrichment data organized by modification type groups.
	Creates separate horizontal groupings for each modification type (Acetylation, Methylation, etc.)
	while maintaining the three-row metric structure.
	"""
	
	def __init__(self, enrichment_results, n):
		"""
		Initialize the plotter with enrichment results data.
		
		Parameters:
		-----------
		enrichment_results : dict
			Dictionary containing 'occupancy', 'entropy', and 'positioning' metrics
			with statistical results for different groups
		subset_n : int
			Number of nucleosomes in the subset for labeling purposes
		"""
		self.enrichment_results = enrichment_results
		self.n = n
		self.organizer = HistoneModificationOrganizer()
		
		# Define modification type groupings
		self.group_keys = ['Methylation', 'Acetylation', 'Phosphorylation', 'Histone Variant']
		self.group_names = ['Methylation', 'Acetylation', 'Phosph.', 'Var.']
		
		# Calculate group counts for proportional sizing
		group_counts_mapping = self.organizer.df.groupby('modification_type').count().rename(
			columns={'modification_name': 'count'})['count']
		self.group_counts = list(group_counts_mapping.loc[self.group_keys].values)
		
	
	def _get_colorbar_params(self, plot_key):
		"""
		Get colorbar parameters for consistent scaling across plots.
		
		Parameters:
		-----------
		plot_key : str
			Which data type ('difference' or 'p_value_fdr')
			
		Returns:
		--------
		dict
			Dictionary containing colormap, vmin, vmax, ticks, labels, and title
		"""
		cmap = get_truncated_RdBu_r()
		
		if plot_key == 'p_value_fdr':
			return {
				'cmap': cmap,
				'vmin': -10,
				'vmax': 10,
				'ticks': [-10, 0, 10],
				'labels': ["$10^{-10}$ depleted", "0", "$10^{-10}$ enriched"],
				'title': 'p-value'
			}
		else:  # difference
			return {
				'cmap': cmap,
				'vmin': -0.25,
				'vmax': 0.25,
				'ticks': [-0.25, 0, 0.25],
				'labels': ['-0.25', '0', '0.25'],
				'title': '$\\Delta$ from\npopulation'
			}

	def plot_colorbar(self, plot_key='difference', figsize=(1.5, 3)):
		"""
		Create a separate figure containing just the colorbar.
		
		Parameters:
		-----------
		plot_key : str
			Which data type to create colorbar for ('difference' or 'p_value_fdr')
		figsize : tuple
			Figure size (width, height) for the colorbar figure
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created colorbar figure
		"""
		# Get colorbar parameters
		cb_params = self._get_colorbar_params(plot_key)
		
		# Create figure and axis for colorbar
		fig, ax = plt.subplots(figsize=figsize)
		
		# Create a mappable object for the colorbar
		sm = plt.cm.ScalarMappable(cmap=cb_params['cmap'], 
								   norm=plt.Normalize(vmin=cb_params['vmin'], 
													  vmax=cb_params['vmax']))
		sm.set_array([])  # Required for ScalarMappable
		
		# Create vertical colorbar
		cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=1.0, pad=0,
			aspect=8.0)
		
		# Configure colorbar
		cbar.ax.set_title(cb_params['title'], pad=15, fontsize=12)
		cbar.ax.set_yticks(cb_params['ticks'])
		cbar.ax.set_yticklabels(cb_params['labels'])
		
		# Hide the main axis since we only want the colorbar
		ax.set_visible(False)
		
		# Adjust layout
		plt.tight_layout()
		
		return fig


	def plot_decile_enrichment_gradient(self, plot_key='difference', n_deciles=10, 
		title=None, figsize=None, metrics=['positioning', 'occupancy', 'entropy'],
		show_metric_label=True):
		"""
		Create a gradient heatmap showing enrichment across all deciles for selected metrics.
		
		Parameters:
		-----------
		plot_key : str
			What to plot ('difference' or 'p_value_fdr')
		n_deciles : int
			Number of deciles to plot (should match analysis)
		title : str, optional
			Custom title for the plot
		figsize : tuple
			Figure size (width, height)
		metrics : list
			List of metrics to plot. Can be any subset of ['positioning', 'occupancy', 'entropy']
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure showing decile gradients
		"""

		if figsize is None:
			if len(metrics) == 3:
				figsize = (13, 13)
				y = 0.98

			elif len(metrics) == 2:
				figsize = (13, 8)
				y = 1.0

			elif len(metrics) == 1:
				figsize = (13, 3.75)
				y = 1.2

		# Create metric name mapping
		metric_display_mapping = {
			'positioning': 'Position',
			'occupancy': 'Occupancy', 
			'entropy': 'Entropy'
		}
		
		# Get display names for selected metrics
		selected_display_names = [metric_display_mapping[metric] for metric in metrics]
		
		# Create proportional subplots with dynamic row count
		fig, axes_groups = create_proportional_subplots(
			self.group_counts,
			n_rows=len(metrics),
			labels=self.group_names,
			figsize=figsize,
			horizontal_padding=0.1,
			vertical_padding=0.75,
			title_rows=[0, 1, 2]  # This may need adjustment based on number of rows
		)
		
		# For each selected metric and modification type group
		for metric_idx, (metric, metric_display_name) in enumerate(zip(metrics, selected_display_names)):
			for group_idx, (modification_type, group_name) in enumerate(zip(self.group_keys, self.group_names)):
				
				ax = axes_groups[metric_idx][group_idx]
				
				# Get modifications for this group
				current_mod_names = self.organizer.get_modifications_flattened_by_group_name(modification_type)
				
				if len(current_mod_names) == 0:
					ax.set_visible(False)
					continue
				
				# Collect data across all deciles for this metric
				decile_data = []
				for decile_num in range(n_deciles):
					decile_key = f'decile_{decile_num}'
					if decile_key in self.enrichment_results[metric]:
						decile_enrichment = self.enrichment_results[metric][decile_key]
						decile_subset = decile_enrichment.set_index('modification').loc[current_mod_names]
						decile_data.append(decile_subset[plot_key].values)
				
				if not decile_data:
					ax.set_visible(False)
					continue
					
				# Create 2D array: rows = deciles, columns = modifications
				plot_matrix = np.array(decile_data)
				
				# Get colorbar parameters for consistent scaling
				cb_params = self._get_colorbar_params(plot_key)
				
				# Create heatmap
				im = ax.imshow(plot_matrix, cmap=cb_params['cmap'], 
							   vmin=cb_params['vmin'], vmax=cb_params['vmax'], 
							   aspect='auto', interpolation='nearest')
				
				# Configure y-axis (deciles)
				if group_idx == 0:
					ax.set_yticks(range(n_deciles))
					ytick_labels = [f'{i+1}' for i in range(n_deciles)]
					ytick_labels = ["Stable, 1"] +  ytick_labels[1:-1] + ["Cyclic, 10"]
					ax.set_yticklabels(ytick_labels, fontsize=14)
				
				# Configure x-axis (modifications)
				show_xticks = True #(metric_idx == len(metrics) - 1)  # Only last selected metric
				if show_xticks:
					ax.set_xticks(range(len(current_mod_names)))

					# Change display name of me to me1 to make clear the distinction of monomethylation
					rename_map = {
						'H3K4me': 'H3K4me1',
						'H3K79me': 'H3K79me1',
					}
					mod_display_names = [rename_map[mod] if mod in rename_map \
						else mod for mod in current_mod_names]

					ax.set_xticklabels(mod_display_names, rotation=45, ha='right',
						fontsize=13)
				else:
					ax.set_xticks([])

				ax.set_ylim(-0.5, plot_matrix.shape[0]-0.5)
				
				# Configure y-axis label
				if show_metric_label:
					show_ylabel = (group_idx == 0)  # Only first column
					if show_ylabel and metric_display_name:
						ax.set_ylabel(metric_display_name, rotation=0, ha='right', va='center', fontsize=20)
				
				# Add significance markers if plotting differences
				if plot_key == 'difference':
					sig = 1e-3
					super_sig = 1e-5
					for decile_idx in range(n_deciles):
						decile_key = f'decile_{decile_idx}'
						if decile_key in self.enrichment_results[metric]:
							decile_enrichment = self.enrichment_results[metric][decile_key]
							decile_subset = decile_enrichment.set_index('modification').loc[current_mod_names]
							for mod_idx, (name, row) in enumerate(decile_subset.iterrows()):
								if row.p_value_fdr < sig:
									sig_text = '*'
									if row.p_value_fdr < super_sig:
										sig_text = '**'
									ax.text(mod_idx, decile_idx, sig_text, ha='center', va='center', 
										   fontsize=8, color='black')
				
				# Add modification type separators
				# for histones (e.g. H2, H3, H4)
				histone_separators = {
					'Acetylation': [1, 8],
					'Methylation': [8], 
					'Phosphorylation': [1],
				}

				residue_separators = {
					'Methylation': [3, 6, 9], 
				}

				if modification_type in residue_separators.keys():
					for index in residue_separators[modification_type]:
						ax.axvline(index-0.5, c='black', ls='solid', zorder=1, lw=.5)
				
				if modification_type in histone_separators.keys():
					for index in histone_separators[modification_type]:
						ax.axvline(index-0.5, c='black', zorder=1, lw=1.5)

				def _format_axes(ax):
					spine_border_width = 1.5
					ax.spines['top'].set_linewidth(spine_border_width)
					ax.spines['bottom'].set_linewidth(spine_border_width)
					ax.spines['left'].set_linewidth(spine_border_width)
					ax.spines['right'].set_linewidth(spine_border_width)

				_format_axes(ax)
		
		# Add overall title
		title_label = "enrichment gradient" if plot_key == 'p_value_fdr' else "difference gradient"
		
		if title is None:
			title = f"Histone modification across cyclic PTR deciles, n={self.n}"

		fig.suptitle(title, fontsize=24, fontweight='demi', y=y)
		
		return fig
