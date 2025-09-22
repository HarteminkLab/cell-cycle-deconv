import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.plot_helpers import create_proportional_subplots_3rows, get_truncated_RdBu_r
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
		self.group_keys = ['Acetylation', 'Methylation', 'Phosphorylation', 'Histone Variant']
		self.group_names = ['Acetylation', 'Methylation', 'Phosph.', 'Var.']
		
		# Calculate group counts for proportional sizing
		group_counts_mapping = self.organizer.df.groupby('modification_type').count().rename(
			columns={'modification_name': 'count'})['count']
		self.group_counts = list(group_counts_mapping.loc[self.group_keys].values)
		
	def _get_group_modifications(self, modification_type):
		"""
		Get modifications for a specific modification type, ordered correctly.
		
		Parameters:
		-----------
		modification_type : str
			The modification type to filter by
			
		Returns:
		--------
		list
			Ordered list of modification names for this type
		"""
		group_mods = self.organizer.get_by_modification_type(modification_type).modification_name.values
		# Apply the global histones ordering but filter to only this group
		all_ordered = histones_ordering()
		return [mod for mod in all_ordered if mod in group_mods]
	
	def _plot_metric_group(self, ax, data_enrichment, modification_type, plot_key='difference', 
						  show_xticks=False, show_ylabel=True, metric_name=""):
		"""
		Plot a single metric for a specific modification type group.
		
		Parameters:
		-----------
		ax : matplotlib.axes.Axes
			The axes to plot on
		data_enrichment : DataFrame
			Enrichment data for this metric
		modification_type : str
			The modification type to plot
		plot_key : str
			Which data to plot ('difference' or 'p_value_fdr')
		show_xticks : bool
			Whether to show x-axis tick labels
		show_ylabel : bool
			Whether to show y-axis label
		metric_name : str
			Name of the metric for labeling
			
		Returns:
		--------
		matplotlib.image.AxesImage
			The image object for colorbar creation
		"""
		# Get modifications for this group
		current_mod_names = self._get_group_modifications(modification_type)
		
		if len(current_mod_names) == 0:
			# Handle empty groups
			ax.set_visible(False)
			return None
			
		# Filter and order data
		data_subset = data_enrichment.set_index('modification').loc[current_mod_names]
		
		# Prepare plot values
		if plot_key == 'p_value_fdr':
			p_val = data_subset.p_value_fdr.values
			logpval = -np.log10(p_val)
			sign = np.sign(data_subset.difference).values
			plot_values = np.multiply(logpval, sign)[None, :]
			vmax = 10
		else:
			plot_values = data_subset[plot_key].values[None, :]
			vmax = 0.25
		
		# Get colorbar parameters for consistent scaling
		cb_params = self._get_colorbar_params(plot_key)

		# Create heatmap using consistent parameters
		im = ax.imshow(plot_values, cmap=cb_params['cmap'], 
					   vmin=cb_params['vmin'], vmax=cb_params['vmax'], aspect='auto')
		
		# Configure axes
		ax.set_yticks([])
		if show_ylabel and metric_name:
			ax.set_ylabel(metric_name, rotation=0, ha='right', va='center', fontsize=14,
				labelpad=0)
		
		# Add significance markers if plotting differences
		if plot_key == 'difference':
			sig = 1e-3
			super_sig = 1e-5
			for index, (name, row) in enumerate(data_subset.iterrows()):
				if row.p_value_fdr < sig:
					sig_text = '*'
					if row.p_value_fdr < super_sig:
						sig_text = '**'
					ax.text(index, 0, sig_text, ha='center', va='center', fontsize=9)

		# Hard-coded separation of histone groupings per modification type
		horizontal_seps = {
			'Acetylation': [1, 8],
			'Methylation': [8],
			'Phosphorylation': [1],
		}

		if modification_type in horizontal_seps.keys():
			for index in horizontal_seps[modification_type]:
				ax.axvline(index-0.5, c='black', zorder=1, lw=.75)
		
		# Configure x-axis
		if show_xticks:
			xticks = np.arange(len(data_subset))
			xtick_labels = data_subset.index
			ax.set_xticks(xticks)
			ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=13)
		else:
			ax.set_xticks([])
		
		return im
	
	def plot_grouped_enrichment(self, group='high', plot_key='difference', figsize=(13, 2.5),
		title=None):
		"""
		Create the main grouped enrichment plot.
		
		Parameters:
		-----------
		group : str
			Which group to plot ('high' for cycling, other for random)
		plot_key : str
			What to plot ('difference' or 'p_value_fdr')
		figsize : tuple
			Figure size (width, height)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create proportional subplots
		fig, axes_groups = create_proportional_subplots_3rows(
			self.group_counts,
			labels=self.group_names,
			figsize=figsize,
			horizontal_padding=0.1,
			vertical_padding=0.2,
			title_rows=[0, 1, 2]
		)
		
		# Get enrichment data
		pos_enrichment = self.enrichment_results['positioning'][group]
		occ_enrichment = self.enrichment_results['occupancy'][group]
		entropy_enrichment = self.enrichment_results['entropy'][group]
		
		# Create metric names
		if group == 'high':
			suffix = "cyclers"
		elif group == 'low':
			suffix = "non-cyclers"
		else:
			suffix = "(random)"

		metric_names = [
			f"Position {suffix},\nn={self.subset_n}",
			f"Occupancy {suffix},\nn={self.subset_n}",
			f"Entropy {suffix},\nn={self.subset_n}"
		]
		
		# Plot each group and metric combination
		im = None  # Keep reference to last image for colorbar
		for group_idx, (modification_type, group_name) in enumerate(zip(self.group_keys, self.group_names)):
			for metric_idx, (enrichment_data, metric_name) in enumerate(zip(
				[pos_enrichment, occ_enrichment, entropy_enrichment], metric_names)):
				
				ax = axes_groups[metric_idx][group_idx]
				show_xticks = (metric_idx == 2)  # Only bottom row shows x-ticks
				show_ylabel = (group_idx == 0)  # Only first column shows y-labels

				current_im = self._plot_metric_group(
					ax, enrichment_data, modification_type, 
					plot_key=plot_key, show_xticks=show_xticks, 
					show_ylabel=show_ylabel, metric_name=metric_name if show_ylabel else ""
				)

		
		# Add overall title
		if group == 'high':
			titlesuffix = "highest cyclers"
		elif group == 'low':
			titlesuffix = "lowest cyclers"
		else:
			 titlesuffix = 'random nucleosomes'

		title_label = "enrichment" if plot_key == 'p_value_fdr' else "differences"

		if title is None:
			title = f"Histone modification {title_label}, {titlesuffix}"
		fig.suptitle(title, 
					fontsize=24, fontweight='demi')
		
		return fig

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

	def plot_all_variants(self, figsize=(13, 8)):
		"""
		Create a comprehensive plot showing both difference and p-value plots for cycling nucleosomes.
		
		Parameters:
		-----------
		figsize : tuple
			Figure size (width, height)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure with both plot types
		"""
		fig = plt.figure(figsize=figsize)
		
		# Create two subplot areas
		gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
		
		# Plot differences (top)
		plt.subplot(gs[0])
		self.plot_grouped_enrichment(group='high', plot_key='difference', figsize=figsize)
		
		# Plot p-values (bottom)  
		plt.subplot(gs[1])
		self.plot_grouped_enrichment(group='high', plot_key='p_value_fdr', figsize=figsize)
		
		return fig


	def plot_decile_enrichment_gradient(self, plot_key='difference', n_deciles=10, 
		title=None,
		figsize=(13, 13)):
		"""
		Create a gradient heatmap showing enrichment across all deciles for all metrics.
		
		Parameters:
		-----------
		plot_key : str
			What to plot ('difference' or 'p_value_fdr')
		n_deciles : int
			Number of deciles to plot (should match analysis)
		figsize : tuple
			Figure size (width, height)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure showing decile gradients
		"""
		# Create proportional subplots
		fig, axes_groups = create_proportional_subplots_3rows(
			self.group_counts,
			labels=self.group_names,
			figsize=figsize,
			horizontal_padding=0.1,
			vertical_padding=0.5,
			title_rows=[0, 1, 2]
		)
		
		# Metric names and data
		metrics = ['positioning', 'occupancy', 'entropy']
		metric_names = [
			f"Position",
			f"Occupancy", 
			f"Entropy"
		]
		
		# For each metric and modification type group
		for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
			for group_idx, (modification_type, group_name) in enumerate(zip(self.group_keys, self.group_names)):
				
				ax = axes_groups[metric_idx][group_idx]
				
				# Get modifications for this group
				current_mod_names = self._get_group_modifications(modification_type)
				
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
					ytick_labels = ["Cyclic, 1"] +  ytick_labels[1:-1] + ["Stable, 10"]
					ax.set_yticklabels(ytick_labels, fontsize=13)
				
				# Configure x-axis (modifications)
				show_xticks = True#(metric_idx == 2)  # Only bottom row
				if show_xticks:
					ax.set_xticks(range(len(current_mod_names)))
					ax.set_xticklabels(current_mod_names, rotation=45, ha='right')
				else:
					ax.set_xticks([])
				
				# Configure y-axis label
				show_ylabel = (group_idx == 0)  # Only first column
				if show_ylabel and metric_name:
					ax.set_ylabel(metric_name, rotation=0, ha='right', va='center', fontsize=20)
				
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
				horizontal_seps = {
					'Acetylation': [1, 8],
					'Methylation': [8], 
					'Phosphorylation': [1],
				}
				
				if modification_type in horizontal_seps.keys():
					for index in horizontal_seps[modification_type]:
						ax.axvline(index-0.5, c='black', zorder=1, lw=.75)
		
		# Add overall title
		title_label = "enrichment gradient" if plot_key == 'p_value_fdr' else "difference gradient"
		
		if title is None:
			title = f"Histone modification across cyclic PTR deciles, n={self.n}"

		fig.suptitle(title, 
					fontsize=24, fontweight='demi', y=0.98)
		
		return fig