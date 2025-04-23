
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline.expression_chromatin_analysis_runner import ExpressionChromatinAnalysis
from src.plot_helpers import adjust_lightness_saturation


class ChromatinCopyNumberAnalysis:
	"""
	A class for analyzing and comparing chromatin data with and without copy number correction.
	Facilitates the analysis of how copy correction affects chromatin dynamics
	in mother and daughter cells during yeast cell cycle.
	"""
	
	def __init__(self, output_dir):
		"""
		Initialize the analysis object.
		
		Parameters:
		-----------
		output_dir : str
			Base directory for input/output files
		"""
		self.output_dir = output_dir
		self.analysis_with_copy = None
		self.analysis_no_copy = None

	def initialize_replication_time_colormaps(self):

		import matplotlib as mpl
		norm = mpl.colors.Normalize(vmin=30, vmax=60)
		cmap = plt.cm.RdBu  # The _r suffix reverses the colormap

		# Create a ScalarMappable object with the colormap
		self.repl_sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		self.repl_cmap = cmap
		self.repl_norm = norm

	def plot_colorbar(self):
		width, height = 0.5, 2
		sm = self.repl_sm
		sm.set_array([])

		# Create a new figure for the colorbar
		fig = plt.figure(figsize=(width, height))

		ax = fig.add_axes([0.3, 0.05, 0.3, 0.9])

		# Create the colorbar
		cbar = plt.colorbar(sm, cax=ax)
		cbar.set_label('Replication time, min', rotation=270, va='bottom')

		
	def initialize_analyses(self):
		"""
		Initialize both copy-corrected and non-copy-corrected analyses.
		"""
		print("Initializing ChromatinCopyNumberAnalysis...")
		
		# Initialize with copy correction
		self.analysis_with_copy = ExpressionChromatinAnalysis(self.output_dir)
		self.analysis_with_copy.initialize_analyses(copy_correction=True)
		
		# Initialize without copy correction
		self.analysis_no_copy = ExpressionChromatinAnalysis(self.output_dir)
		self.analysis_no_copy.initialize_analyses(copy_correction=False)
		
	def load_data(self):
		"""
		Load all required data for both analyses including polar datasets
		and gene replication times.
		"""
		print("Loading data for analysis...")
		
		# Create polar datasets
		self.analysis_with_copy.create_polar_datasets()
		self.analysis_no_copy.create_polar_datasets()
		
		# Load gene replication times
		self.analysis_with_copy.load_gene_replication_times()
		self.analysis_no_copy.load_gene_replication_times()
		
		print("Data loading complete.")
	
	def prepare_dataset_pair(self, no_copy_df, w_copy_df):
		"""
		Prepare a matching pair of datasets for comparison.
		
		Parameters:
		-----------
		no_copy_df : DataFrame
			DataFrame without copy correction
		w_copy_df : DataFrame
			DataFrame with copy correction
			
		Returns:
		--------
		tuple: (no_copy_set, w_copy_set)
			Matched DataFrames for comparison
		"""
		no_copy_set = no_copy_df.dropna()
		w_copy_set = w_copy_df.loc[no_copy_set.index]
		return no_copy_set, w_copy_set
	
	def create_comparison_subplot(self, ax, no_copy_set, w_copy_set, ptr_column, title,
								lims=(0.9, 2.3)):
		"""
		Create a scatter plot comparing PTR values with and without copy correction.
		
		Parameters:
		-----------
		ax : matplotlib.axes.Axes
			The axis to plot on
		no_copy_set : DataFrame
			Dataset without copy correction
		w_copy_set : DataFrame
			Dataset with copy correction
		ptr_column : str
			Column name for the PTR values
		title : str
			Title for the subplot
		lims : tuple, optional
			Axis limits, default (0.9, 2.3)
		"""
		replication_times = self.analysis_with_copy.gene_replication_times.loc[no_copy_set.index].\
			replication_time
		
		ax.scatter(no_copy_set[ptr_column], w_copy_set[ptr_column],
				  c=replication_times,
				  cmap=self.repl_cmap, norm=self.repl_norm, s=5)
		ax.set_xlabel("No copy correction PTR")
		ax.set_ylabel("Copy corrected PTR")
		ax.plot([0, 10], [0, 10], c='black', lw=1, ls='dotted')
		ax.set_xlim(*lims)
		ax.set_ylim(*lims)
		ax.set_title(title)
	
	def create_comparison_figure(self, figsize=(11, 7), mother_orfs=None, daughter_orfs=None,
		title=None):
		"""
		Create a full figure with 6 subplots comparing various datasets.
		
		Parameters:
		-----------
		figsize : tuple, optional
			Figure size, default (11, 7)
		orfs : list or pandas.Index, optional
			Specific ORFs to include in the analysis
		title : str, optional
			Figure title
			
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The created figure
		"""
		
		# Prepare datasets
		datasets = {
			'promoter': self.prepare_dataset_pair(
				self.analysis_no_copy.small_combined_polar_data_df,
				self.analysis_with_copy.small_combined_polar_data_df
			),
			'entropy': self.prepare_dataset_pair(
				self.analysis_no_copy.entropies_combined_polar_data_df,
				self.analysis_with_copy.entropies_combined_polar_data_df
			),
			'gene_body': self.prepare_dataset_pair(
				self.analysis_no_copy.gb_nucleosome_combined_polar_data_df,
				self.analysis_with_copy.gb_nucleosome_combined_polar_data_df
			)
		}
		
		# Create figure and axes
		fig, axes = plt.subplots(2, 3, figsize=figsize)
		
		# Flatten axes for easier indexing
		axes = np.array(axes).T.flatten()
		
		# Plot configurations
		default_ylims = (0.9, 2.7)
		plot_configs = [
			# (index, dataset_key, ptr_column, title)
			(0, 'promoter', 'ptr_t', "mother", "promoter change", default_ylims),
			(1, 'promoter', 'ptr_b', "daughter", "promoter change", default_ylims),
			(2, 'gene_body', 'ptr_t', "mother", "gene body change", default_ylims),
			(3, 'gene_body', 'ptr_b', "daughter", "gene body change", default_ylims),
			(4, 'entropy', 'ptr_t', "mother", "entropy change", default_ylims),
			(5, 'entropy', 'ptr_b', "daughter", "entropy change", default_ylims),
		]

		def subset_df_common_orfs(df, orfs):
			return df.loc[list(set(df.index.values).intersection(orfs))]
		
		# Create subplots
		for idx, dataset_key, ptr_column, branch, metric, lims in plot_configs:
			no_copy_set, w_copy_set = datasets[dataset_key]

			if mother_orfs is not None and daughter_orfs is not None:
				if branch == 'mother':
					no_copy_set = subset_df_common_orfs(no_copy_set, mother_orfs)
					w_copy_set = subset_df_common_orfs(w_copy_set, mother_orfs)
				elif branch == 'daughter':
					no_copy_set = subset_df_common_orfs(no_copy_set, daughter_orfs)
					w_copy_set = subset_df_common_orfs(w_copy_set, daughter_orfs)

			title_text = f"{branch.title()}, {metric}"
			self.create_comparison_subplot(axes[idx], 
				no_copy_set, w_copy_set, ptr_column, title_text, lims)
		
		plt.tight_layout()
		plt.subplots_adjust(bottom=0.15, hspace=0.35, wspace=0.25)
		
		if title:
			plt.suptitle(title, y=1.05)
		
		return fig
	
	def get_expression_segments(self, quantiles, key):
		"""
		Segment genes by expression level.
		
		Parameters:
		-----------
		quantiles : list, optional
			Quantile cutoffs, default [0.2, 0.4, 0.6, 0.8]
			
		Returns:
		--------
		tuple: (segments, quantile_values, segment_lengths)
			Segments of genes by expression level
		"""
		from src.helpers import get_quantile_values
		
		mean_expression = self.analysis_with_copy.entropies_combined_polar_data_df[key]
		segments, qvals, lens = get_quantile_values(mean_expression, q=quantiles)
		
		return segments, qvals, lens
	
	def plot_expression_distribution(self, quantiles=[0.2, 0.4, 0.6, 0.8], figsize=(9, 3)):
		"""
		Plot the distribution of expression values with quantile lines.
		
		Parameters:
		-----------
		quantiles : list, optional
			Quantile cutoffs, default [0.2, 0.4, 0.6, 0.8]
		figsize : tuple, optional
			Figure size, default (10, 6)
			
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The created figure
		"""
		fig, (ax_m, ax_d) = plt.subplots(1, 2, figsize=figsize)

		def plot_dist(ax, mean_branch_key):
		
			mean_expression = self.analysis_with_copy.entropies_combined_polar_data_df[mean_branch_key]
			ax.hist(mean_expression, bins=np.linspace(2, 6, 50))
			ax.set_title("Expression Distribution")
			ax.set_xlabel("Mean Expression")
			ax.set_ylabel("Count")
			
			# Mother branch distribution
			segments, qvals, lens = self.get_expression_segments(quantiles, mean_branch_key)
			for q in qvals:
				ax.axvline(q, c='red', alpha=0.75)

		plot_dist(ax_m, 'mean_t')
		plt.title('Mother')

		plot_dist(ax_d, 'mean_t')
		plt.title('Daughter')
		
		plt.tight_layout()

		return fig
	
	def create_expression_stratified_plots(self, figsize=(11, 7)):
		"""
		Create plots stratified by expression level.
		
		Parameters:
		-----------
		figsize : tuple, optional
			Figure size for each plot, default (11, 7)
			
		Returns:
		--------
		dict : Dictionary of matplotlib figures
			The created figures keyed by expression level
		"""
		mother_segments, _, _ = self.get_expression_segments(quantiles=[0.333, 0.667], key='mean_t')
		daughter_segments, _, _ = self.get_expression_segments(quantiles=[0.333, 0.667], key='mean_b')
		
		figures = {}
		# Plot for lowest expression segment
		figures['low'] = self.create_comparison_figure(
			figsize=figsize,
			mother_orfs=mother_segments[0].index,
			daughter_orfs=daughter_segments[0].index,
			title="Low mean expression"
		)
		
		# Plot for highest expression segment
		figures['high'] = self.create_comparison_figure(
			figsize=figsize,
			mother_orfs=mother_segments[-1].index,
			daughter_orfs=daughter_segments[-1].index,
			title="High mean expression"
		)
		
		return figures
			
	def calculate_delta_ptr_df(self, delta_func='difference'):
		"""
		Create a dataframe containing delta PTR values (copy corrected - uncorrected)
		for all chromatin features in mother and daughter cells.
		
		Returns:
		--------
		DataFrame
			Contains delta PTR values for all genes and features
		"""
		# Get common genes across all datasets
		common_genes = (self.analysis_with_copy.small_combined_polar_data_df.index
					   .intersection(self.analysis_no_copy.small_combined_polar_data_df.index)
					   .intersection(self.analysis_with_copy.entropies_combined_polar_data_df.index)
					   .intersection(self.analysis_with_copy.gb_nucleosome_combined_polar_data_df.index))
		
		# Create dataframe to store results
		delta_ptr_df = pd.DataFrame(index=common_genes)
		
		# Add replication timing
		delta_ptr_df['replication_time'] = \
			self.analysis_with_copy.gene_replication_times.loc[common_genes].replication_time
		
		# Add mean expression
		delta_ptr_df['mean_expression'] = \
			self.analysis_with_copy.entropies_combined_polar_data_df.loc[common_genes, 'mean_t']
		
		# Calculate delta PTRs for each feature
		features = {
			'promoter': (
				self.analysis_with_copy.small_combined_polar_data_df,
				self.analysis_no_copy.small_combined_polar_data_df
			),
			'gene_body': (
				self.analysis_with_copy.gb_nucleosome_combined_polar_data_df,
				self.analysis_no_copy.gb_nucleosome_combined_polar_data_df
			),
			'entropy': (
				self.analysis_with_copy.entropies_combined_polar_data_df,
				self.analysis_no_copy.entropies_combined_polar_data_df
			)
		}
		
		for feature_name, (with_copy_df, no_copy_df) in features.items():

			if delta_func == 'difference':
				# Mother delta PTR
				delta_ptr_df[f'{feature_name}_delta_ptr_mother'] = (
					with_copy_df.loc[common_genes, 'ptr_t'] - 
					no_copy_df.loc[common_genes, 'ptr_t']
				)
				
				# Daughter delta PTR
				delta_ptr_df[f'{feature_name}_delta_ptr_daughter'] = (
					with_copy_df.loc[common_genes, 'ptr_b'] - 
					no_copy_df.loc[common_genes, 'ptr_b']
				)
			elif delta_func == 'logratio':

				eps = 1e-5
				# Mother delta PTR
				delta_ptr_df[f'{feature_name}_delta_ptr_mother'] = np.log2(
					(with_copy_df.loc[common_genes, 'ptr_t']+eps) /
					(no_copy_df.loc[common_genes, 'ptr_t']+eps)
				)
				
				# Daughter delta PTR
				delta_ptr_df[f'{feature_name}_delta_ptr_daughter'] = np.log2(
					(with_copy_df.loc[common_genes, 'ptr_b']+eps) /
					(no_copy_df.loc[common_genes, 'ptr_b']+eps)
				)
		
		return delta_ptr_df

	def get_replication_timing_segments(self, timing_quantile_thresholds=[0.333, 0.667]):
		"""
		Segment genes by replication timing.
		
		Parameters:
		-----------
		timing_thresholds : list, optional
			Thresholds for defining early, mid, and late replication
			
		Returns:
		--------
		dict
			Dictionary mapping 'early', 'mid', 'late' to indices of genes
		"""
		delta_ptr_df = self.calculate_delta_ptr_df()
		replication_times = delta_ptr_df['replication_time']

		from src.helpers import get_quantile_values
		segments, qvals, lens = get_quantile_values(replication_times, q=timing_quantile_thresholds)
		
		segments = {
			'early': segments[0].index,
			'mid': segments[1].index,
			'late': segments[2].index
		}
		
		# Define colors for each segment
		self.replication_colors = {
			'early': plt.cm.RdBu(0.2),
			'mid': plt.cm.Greys(0.5),
			'late': plt.cm.RdBu(0.8)
		}
		
		return segments

	def prepare_delta_ptr_data(self, feature):
		"""
		Prepare a dataframe containing delta PTR data with expression and replication timing groups,
		separating expression segments by mother and daughter branches.
		
		Parameters:
		-----------
		feature : str
			Chromatin feature ('promoter', 'gene_body', or 'entropy')
			
		Returns:
		--------
		pandas.DataFrame
			DataFrame with columns for gene, mother_expression_group, daughter_expression_group, 
			replication_group, delta_ptr_mother, delta_ptr_daughter
		"""
		# Get delta PTR dataframe
		delta_ptr_df = self.calculate_delta_ptr_df(delta_func='difference')
		
		# Get expression segments separately for mother and daughter branches
		mother_expression_segments, mother_qvals, _ = self.get_expression_segments(
			quantiles=[0.333, 0.667], key='mean_t')
		daughter_expression_segments, daughter_qvals, _ = self.get_expression_segments(
			quantiles=[0.333, 0.667], key='mean_b')
		expression_labels = ['Low', 'Medium', 'High']
		
		# Get replication timing segments
		rep_segments = self.get_replication_timing_segments()
		
		# Initialize the data list to store rows for the DataFrame
		data_rows = []
		
		# Get all genes in delta_ptr_df
		all_genes = delta_ptr_df.index
		
		# For each gene, determine its expression groups and replication timing
		for gene in all_genes:
			# Get replication timing group
			rep_group = None
			for timing in ['early', 'mid', 'late']:
				if gene in rep_segments[timing]:
					rep_group = timing
					break
			
			if rep_group is None:
				continue  # Skip genes without replication timing data
			
			# Get mother expression group
			mother_expr_group = None
			for i, segment in enumerate(mother_expression_segments):
				if gene in segment.index:
					mother_expr_group = expression_labels[i]
					break
			
			# Get daughter expression group
			daughter_expr_group = None
			for i, segment in enumerate(daughter_expression_segments):
				if gene in segment.index:
					daughter_expr_group = expression_labels[i]
					break
			
			# Get delta PTR values
			mother_delta = delta_ptr_df.loc[gene, f'{feature}_delta_ptr_mother'] \
				if gene in delta_ptr_df.index else None
			daughter_delta = delta_ptr_df.loc[gene, f'{feature}_delta_ptr_daughter'] \
				if gene in delta_ptr_df.index else None
			
			# Add to data rows if we have valid expression groups and at least one valid delta PTR value
			if (mother_expr_group or daughter_expr_group) and (pd.notna(mother_delta) or pd.notna(daughter_delta)):
				data_rows.append({
					'gene': gene,
					'mother_expression_group': mother_expr_group,
					'daughter_expression_group': daughter_expr_group,
					'replication_group': rep_group,
					'delta_ptr_mother': mother_delta,
					'delta_ptr_daughter': daughter_delta
				})
		
		# Create DataFrame from the data rows
		result_df = pd.DataFrame(data_rows)
		
		return result_df

	def plot_delta_ptr_violins_from_df(self, data_df, feature, figsize=(11, 5), ylims=None):
		"""
		Create violin plots showing the delta PTR distributions using the prepared dataframe.
		
		Parameters:
		-----------
		data_df : pandas.DataFrame
			DataFrame containing the delta PTR data with branch-specific expression and replication groups
		feature : str
			Chromatin feature to plot ('promoter', 'gene_body', or 'entropy')
		figsize : tuple
			Figure size
		ylims : tuple or None
			Y-axis limits
				
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		feature_titles = {
			'promoter': "Promoter occupancy",
			'gene_body': "Gene body nucleosome occupancy",
			'entropy': "Gene body nucleosome entropy"
		}
		
		# Create figure
		fig, (ax_mother, ax_daughter) = plt.subplots(1, 2, figsize=figsize, sharey=True)
		
		# Define expression and replication groups
		expression_labels = ['Low', 'Medium', 'High']
		replication_labels = ['early', 'mid', 'late']
		
		positions = []
		current_pos = 0
		group_width = 3.5
		segment_width = 0.75
		
		# Track violin plots for legend
		violin_plots = []
		
		# Loop through expression segments for mother
		for i, expr_label in enumerate(expression_labels):
			# Track positions for this expression group
			group_positions = []
			
			# For each replication timing group
			for j, timing in enumerate(replication_labels):
				# Filter data for mother branch with this expression and replication group
				mother_filter = (data_df['mother_expression_group'] == expr_label) & \
					(data_df['replication_group'] == timing)
				mother_values = data_df.loc[mother_filter, 'delta_ptr_mother'].dropna()
				
				# Position for this violin
				pos = current_pos + j * segment_width
				group_positions.append(pos)
				
				# Calculate mean for diamond marker
				mother_mean = mother_values.mean() if not mother_values.empty else np.nan

				bw = 0.2
				
				# Plot violin for mother if we have data
				if not mother_values.empty:
					vp_mother = ax_mother.violinplot(
						mother_values,
						positions=[pos],
						showmeans=False,
						showmedians=True,
						points=500,
						bw_method=bw,
					)
					
					# Add diamond marker for mean
					if not np.isnan(mother_mean):
						ax_mother.scatter([pos], [mother_mean], color='black', marker='D', s=20, zorder=3)

					# Set violin colors based on replication timing
					for partname in ('bodies', 'cmedians', 'cmaxes', 'cmins', 'cbars'):
						if partname in vp_mother:
							color = self.replication_colors[timing]

							if type(vp_mother[partname]) == list:
								for part in vp_mother[partname]:
									part.set_color(color)
							else:
								vp_mother[partname].set_color(color)
					
					# Save for legend (only on first expression group)
					if i == 0:
						violin_plots.append((vp_mother, timing))
			
			# Move to next expression group
			positions.append(group_positions)
			current_pos += group_width
		
		# Reset positions for daughter plot
		positions = []
		current_pos = 0
		
		# Loop through expression segments for daughter
		for i, expr_label in enumerate(expression_labels):
			# Track positions for this expression group
			group_positions = []
			
			# For each replication timing group
			for j, timing in enumerate(replication_labels):
				# Filter data for daughter branch with this expression and replication group
				daughter_filter = (data_df['daughter_expression_group'] == expr_label) & \
					(data_df['replication_group'] == timing)
				daughter_values = data_df.loc[daughter_filter, 'delta_ptr_daughter'].dropna()
				
				# Position for this violin
				pos = current_pos + j * segment_width
				group_positions.append(pos)
				
				# Calculate mean for diamond marker
				daughter_mean = daughter_values.mean() if not daughter_values.empty else np.nan

				bw = 0.2
				
				# Plot violin for daughter if we have data
				if not daughter_values.empty:
					vp_daughter = ax_daughter.violinplot(
						daughter_values,
						positions=[pos],
						showmeans=False,
						showmedians=True,
						points=500,
						bw_method=bw,
					)
					
					# Add diamond marker for mean
					if not np.isnan(daughter_mean):
						ax_daughter.scatter([pos], [daughter_mean], color='black', marker='D', s=20, zorder=3)

					# Set violin colors based on replication timing
					for partname in ('bodies', 'cmedians', 'cmaxes', 'cmins', 'cbars'):
						if partname in vp_daughter:
							color = self.replication_colors[timing]

							if type(vp_daughter[partname]) == list:
								for part in vp_daughter[partname]:
									part.set_color(color)
							else:
								vp_daughter[partname].set_color(color)
			
			# Move to next expression group
			positions.append(group_positions)
			current_pos += group_width
		
		# Set axis properties
		for ax, title, expr_group_col in zip(
			[ax_mother, ax_daughter], 
			['Mother', 'Daughter'],
			['mother_expression_group', 'daughter_expression_group']
		):
			# Set title and labels
			ax.set_title(f"{title} branch", fontsize=14, y=1.03)

			if title == 'Mother':
				ax.set_ylabel('Δ PTR, copy corrected - uncorrected', fontsize=12)

			ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
			ax.set_xlabel(f"Expression level", fontsize=12)
			
			# Set x-ticks at center of each expression group
			group_centers = [sum(group)/len(group) if group else i*group_width+group_width/2 \
				for i, group in enumerate(positions)]
			ax.set_xticks(group_centers)
			ax.set_xticklabels(expression_labels, fontsize=10)
			
			# Add grid
			ax.grid(axis='y', linestyle='-', lw=0.25)

			if ylims is not None:
				ax.set_ylim(*ylims)
		
		# Add overall title
		feature_title = feature_titles[feature].lower()
		plt.suptitle(f'Copy correction Δ PTR for {feature_title}', 
					 fontsize=18, fontweight='demi', y=1)
		
		plt.tight_layout()
		plt.subplots_adjust(wspace=0.12)
		return fig

	def create_repl_groups_legend(self):

		from src.plot_helpers import hide_spines

		replication_groups = ['early', 'mid', 'late']
		legend_elements = [plt.Line2D([0], [0], color=self.replication_colors[timing], 
									 lw=4) for timing in replication_groups]


		labels = [f"{timing.title()} replication" for timing in replication_groups]

		plt.figure(figsize=(5, 1))
		ax = plt.gca()
		legend = plt.legend(legend_elements, labels, loc=(0.1, 0))
		ax.add_artist(legend)
		hide_spines(ax)


	def create_delta_ptr_violin_plots_with_df(self, feature, figsize=(11, 5), ylims=None):
		"""
		Create violin plots showing the delta PTR distributions and return the data dataframe.
		
		Parameters:
		-----------
		feature : str
			Chromatin feature to plot ('promoter', 'gene_body', or 'entropy')
		figsize : tuple
			Figure size
				
		Returns:
		--------
		tuple
			(matplotlib.figure.Figure, pandas.DataFrame)
			The created figure and the underlying data dataframe
		"""
		# Prepare the data
		data_df = self.prepare_delta_ptr_data(feature)
		
		# Create the plot
		fig = self.plot_delta_ptr_violins_from_df(data_df, feature, figsize, ylims)
		
		# Return both the figure and the dataframe
		return fig, data_df

	def plot_ptr_violin_plots_by_expression(self, feature, branch_expr_selection=None, 
		figsize=(9, 6), ylims=None, title=None, sup_y_adjust=None):
		"""
		Create violin plots showing the PTR distributions before and after copy number correction,
		stratified by both replication timing and expression level.
		
		Parameters:
		-----------
		feature : str
			Chromatin feature to plot ('promoter', 'gene_body', or 'entropy')
		branch_expr_selection : list of tuples, optional
			List of (branch, expression) tuples to plot. Each tuple should contain:
			- branch: str, 'mother' or 'daughter'
			- expression: str, 'Low', 'Medium', or 'High'
			If None, all combinations will be plotted (2×3 grid)
		figsize : tuple, optional
			Figure size
		ylims : tuple or None
			Y-axis limits
		title : str or None
			Custom title for the plot
		sup_y_adjust : float or None
			Y-position adjustment for the suptitle
			
		Returns:
		--------
		tuple
			(matplotlib.figure.Figure, mother_df, daughter_df, float)
			The created figure, mother and daughter DataFrames, and cell cycling threshold
		"""
		feature_title = feature[0:1].upper() + feature[1:]
		feature_title = feature_title.replace('_', ' ')

		if branch_expr_selection is not None and len(branch_expr_selection) == 2:
			sup_y_adjust=0.96
			figsize = (6, 3)
			title = f"{feature_title} PTR change, {branch_expr_selection[0][1].lower()} expression"
		
		# Get data frames with and without copy correction
		if feature == 'promoter':
			df_with_copy = self.analysis_with_copy.small_combined_polar_data_df
			df_no_copy = self.analysis_no_copy.small_combined_polar_data_df
		elif feature == 'gene_body':
			df_with_copy = self.analysis_with_copy.gb_nucleosome_combined_polar_data_df
			df_no_copy = self.analysis_no_copy.gb_nucleosome_combined_polar_data_df
		elif feature == 'entropy':
			df_with_copy = self.analysis_with_copy.entropies_combined_polar_data_df
			df_no_copy = self.analysis_no_copy.entropies_combined_polar_data_df
		
		# Get common genes
		common_genes = df_with_copy.index.intersection(df_no_copy.index)
		
		# Get expression segments for mother and daughter
		mother_expression_segments, _, _ = self.get_expression_segments(quantiles=[0.333, 0.667], key='mean_t')
		daughter_expression_segments, _, _ = self.get_expression_segments(quantiles=[0.333, 0.667], key='mean_b')
		expression_labels = ['Low', 'Medium', 'High']
		
		# Get replication timing segments
		rep_segments = self.get_replication_timing_segments()
		
		# Define all possible combinations
		all_combinations = [
			('mother', 'Low'), ('mother', 'Medium'), ('mother', 'High'),
			('daughter', 'Low'), ('daughter', 'Medium'), ('daughter', 'High')
		]
		
		# Use specified selection or default to all
		selected_combinations = branch_expr_selection if branch_expr_selection else all_combinations
		
		# Determine grid layout based on number of selected plots
		n_plots = len(selected_combinations)
		if n_plots <= 3:
			n_rows, n_cols = 1, n_plots
		else:
			n_rows = (n_plots + 2) // 3  # Ceiling division
			n_cols = min(3, n_plots)
		
		# Create figure with appropriate grid
		fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
		
		# Handle single subplot case
		if n_plots == 1:
			axes = np.array([axes])
		
		# Flatten axes for easier indexing
		if n_plots > 1:
			axes = np.array(axes).flatten()
		
		# Calculate the cycling threshold
		q_threshold = 0.9
		t_ptrs = df_with_copy['ptr_t'].values
		b_ptrs = df_with_copy['ptr_b'].values
		ptr_values = np.concatenate([t_ptrs, b_ptrs])
		cell_cycling_threshold = np.quantile(ptr_values, q=q_threshold)
		
		# Helper function to set violin colors
		def set_violin_colors(vp, color, alpha=1.0):
			for partname in ('bodies', 'cmedians', 'cmaxes', 'cmins', 'cbars'):
				if partname in vp:
					if type(vp[partname]) == list:
						for part in vp[partname]:
							part.set_color(color)
							part.set_alpha(alpha)
					else:
						vp[partname].set_color(color)
						vp[partname].set_alpha(alpha)
		
		# Create DataFrames to store violin data (instead of nested dictionaries)
		# Initialize empty lists to collect data
		data_rows = []
		
		# Prepare data for all combinations (we'll filter later for plotting)
		for branch in ['mother', 'daughter']:
			branch_key = 'ptr_t' if branch == 'mother' else 'ptr_b'
			expr_segments = mother_expression_segments if branch == 'mother' else daughter_expression_segments
			
			for expr_idx, expr_label in enumerate(expression_labels):
				expr_segment = expr_segments[expr_idx]
				
				for timing in ['early', 'mid', 'late']:
					# Get genes in both the expression segment and replication timing segment
					genes_in_segment = expr_segment.index.intersection(rep_segments[timing])
					genes_in_segment = genes_in_segment.intersection(common_genes)
					
					# Skip if no genes in this segment
					if len(genes_in_segment) == 0:
						continue
					
					# Get PTR values for each gene
					for gene_id in genes_in_segment:
						if gene_id in df_no_copy.index and gene_id in df_with_copy.index:
							no_copy_val = df_no_copy.loc[gene_id, branch_key]
							with_copy_val = df_with_copy.loc[gene_id, branch_key]
							
							if pd.notna(no_copy_val) and pd.notna(with_copy_val):
								data_rows.append({
									'gene_id': gene_id,
									'branch': branch,
									'expression': expr_label,
									'replication': timing,
									'ptr_no_copy': no_copy_val,
									'ptr_with_copy': with_copy_val
								})
		
		# Create one DataFrame with all the violin data
		violin_df = pd.DataFrame(data_rows)
		
		# Split into mother and daughter DataFrames for easier reference
		mother_df = violin_df[violin_df['branch'] == 'mother']
		daughter_df = violin_df[violin_df['branch'] == 'daughter']
		
		# Now we can easily filter this DataFrame for each plot
		legend_plots = []
		
		# Inner function to create violins for a specific branch and expression level
		def create_branch_expr_violins(ax, branch, expr_label, df):
			# Filter the DataFrame for this branch and expression level
			plot_df = df[(df['branch'] == branch) & (df['expression'] == expr_label)]
			
			# Positions for violin plots
			positions = []
			current_pos = 0
			group_width = 2.0
			segment_width = 0.55
			
			# Track violin plots for legend
			violin_plots = []
			
			# For each replication timing group
			for timing in ['early', 'mid', 'late']:
				# Filter for this replication timing
				timing_df = plot_df[plot_df['replication'] == timing]
				
				if len(timing_df) == 0:
					current_pos += group_width
					positions.append((current_pos - group_width, current_pos - group_width + segment_width))
					continue
				
				# Positions for these violins
				pos_no_copy = current_pos
				pos_with_copy = current_pos + segment_width
				positions.append((pos_no_copy, pos_with_copy))
				
				# Get PTR values
				no_copy_values = timing_df['ptr_no_copy'].values
				with_copy_values = timing_df['ptr_with_copy'].values
				
				# Calculate means for diamond markers
				no_copy_mean = np.mean(no_copy_values) if len(no_copy_values) > 0 else np.nan
				with_copy_mean = np.mean(with_copy_values) if len(with_copy_values) > 0 else np.nan
				
				bw = 0.2
				
				# Only plot if we have data
				if len(no_copy_values) > 0 and len(with_copy_values) > 0:
					# Plot violins
					vp_no_copy = ax.violinplot(
						no_copy_values,
						positions=[pos_no_copy],
						showmeans=False,
						showmedians=True,
						points=500,
						bw_method=bw,
					)
					
					vp_with_copy = ax.violinplot(
						with_copy_values,
						positions=[pos_with_copy],
						showmeans=False,
						showmedians=True,
						points=500,
						bw_method=bw,
					)
					
					# Add diamond markers for means
					if not np.isnan(no_copy_mean):
						ax.scatter([pos_no_copy], [no_copy_mean], color='black', marker='D', s=20, zorder=3)
					if not np.isnan(with_copy_mean):
						ax.scatter([pos_with_copy], [with_copy_mean], color='black', marker='D', s=20, zorder=3)
					
					# Set colors
					color = self.replication_colors[timing]
					set_violin_colors(vp_no_copy, color, alpha=0.5)
					set_violin_colors(vp_with_copy, color, alpha=1.0)
					
					# Save for legend
					violin_plots.append((vp_with_copy, timing))
				
				# Move to next replication group
				current_pos += group_width
			
			# Add a horizontal line at the cell cycling threshold
			ax.axhline(y=cell_cycling_threshold, color=plt.cm.Oranges(0.75), 
				linestyle='--', lw=1, alpha=0.7)
			
			# Set axis properties
			ax.set_title(f"{branch.title()} - {expr_label} Expression", fontsize=12)
			ax.set_ylabel('PTR (Peak-to-Trough Ratio)', fontsize=10)
			ax.set_xlabel("Replication timing", fontsize=10)
			
			# Set x-ticks at center of each replication group
			group_centers = [(pos[0] + pos[1])/2 for pos in positions]
			ax.set_xticks(group_centers)
			ax.set_xticklabels(['Early', 'Mid', 'Late'], fontsize=9)
			
			# Add grid
			ax.grid(axis='y', linestyle='-', lw=0.25)
			
			if ylims is not None:
				ax.set_ylim(*ylims)
				
			return violin_plots
		
		# Create violin plots for selected combinations
		for idx, (branch, expr_level) in enumerate(selected_combinations):
			vp = create_branch_expr_violins(axes[idx], branch, expr_level, violin_df)
			
			if idx == 0:  # Only collect legend from first plot
				legend_plots = vp
		
		# Hide empty subplots if any
		for i in range(n_plots, len(axes)):
			axes[i].set_visible(False)
		
		# Add overall title
		if title is None:
			title = f'{feature_title} PTR following copy correction'
		
		if sup_y_adjust is None:
			sup_y_adjust = 0.98
		
		plt.suptitle(title, fontsize=18, fontweight='demi', y=sup_y_adjust)
		
		# Add legend for replication timing
		repl_legend_elements = [plt.Line2D([0], [0], color=self.replication_colors[timing], 
										  lw=4, label=f"{timing.title()} replication") 
							  for _, timing in legend_plots]
		
		# Add legend for copy correction status
		correction_legend_elements = [
			plt.Line2D([0], [0], color='black', lw=2, alpha=0.5, label="No copy correction"),
			plt.Line2D([0], [0], color='black', lw=2, alpha=1.0, label="With copy correction")
		]
		
		plt.tight_layout()
		
		return fig, mother_df, daughter_df, cell_cycling_threshold

	def plot_cycling_threshold_changes(self, feature, mother_df, daughter_df, threshold=None, 
		branch_expr_selection=None, 
		figsize=(9, 6), title=None, sup_y_adjust=None):
		"""
		Create a grid of bar plots showing the number of genes that change cycling status after copy correction,
		organized by branch, expression level, and replication timing.
		
		Parameters:
		-----------
		feature : str
			Chromatin feature being plotted
		mother_df : pandas.DataFrame
			DataFrame containing mother branch data from violin plots
		daughter_df : pandas.DataFrame
			DataFrame containing daughter branch data from violin plots
		threshold : float
			Threshold for 'cell cycling' PTR
		branch_expr_selection : list of tuples, optional
			List of (branch, expression) tuples to plot. Each tuple should contain:
			- branch: str, 'mother' or 'daughter'
			- expression: str, 'Low', 'Medium', or 'High'
			If None, all combinations will be plotted (2×3 grid)
		figsize : tuple, optional
			Figure size
		title : str or None
			Custom title for the plot
		sup_y_adjust : float or None
			Y-position adjustment for the suptitle
			
		Returns:
		--------
		tuple
			(matplotlib.figure.Figure, pandas.DataFrame)
			The created figure and the DataFrame containing the counts data
		"""
		feature_title = feature[0:1].upper() + feature[1:]
		feature_title = feature_title.replace('_', ' ')

		if branch_expr_selection is not None and len(branch_expr_selection) == 2:
			sup_y_adjust=1.22
			figsize = (6, 2.5)
			title = (f"{feature_title} cell cycling\nclass change, "
					 f"{branch_expr_selection[0][1].lower()} expression")

		# Combine mother and daughter DataFrames
		violin_df = pd.concat([mother_df, daughter_df])
		
		# Create a list to store data rows
		data_rows = []
		
		# Process data for each branch, expression level, and replication timing
		for branch in ['mother', 'daughter']:
			branch_df = mother_df if branch == 'mother' else daughter_df
			for expr in ['Low', 'Medium', 'High']:
				for timing in ['early', 'mid', 'late']:
					# Filter the DataFrame for this combination
					filtered_df = branch_df[(branch_df['expression'] == expr) & 
										  (branch_df['replication'] == timing)]
					
					# Skip if no data for this combination
					if len(filtered_df) == 0:
						continue
					
					# Initialize gene lists
					cycle_to_noncycle_genes = []
					noncycle_to_cycle_genes = []
					unchanged_cycling_genes = []
					unchanged_noncycling_genes = []
					
					# Count transitions and collect gene IDs
					for _, row in filtered_df.iterrows():
						before_cycling = row['ptr_no_copy'] >= threshold
						after_cycling = row['ptr_with_copy'] >= threshold
						
						if before_cycling and not after_cycling:
							cycle_to_noncycle_genes.append(row['gene_id'])
						elif not before_cycling and after_cycling:
							noncycle_to_cycle_genes.append(row['gene_id'])
						elif before_cycling and after_cycling:
							unchanged_cycling_genes.append(row['gene_id'])
						else:  # not before_cycling and not after_cycling
							unchanged_noncycling_genes.append(row['gene_id'])
					
					# Add row to data
					data_rows.append({
						'Branch': branch,
						'Expression': expr,
						'Replication': timing,
						'Cycling to Non-cycling': len(cycle_to_noncycle_genes),
						'Non-cycling to Cycling': len(noncycle_to_cycle_genes),
						'Unchanged Cycling': len(unchanged_cycling_genes),
						'Unchanged Non-cycling': len(unchanged_noncycling_genes),
						'Total Genes': len(filtered_df),
						'Cycle_to_Noncycle_Genes': cycle_to_noncycle_genes,
						'Noncycle_to_Cycle_Genes': noncycle_to_cycle_genes,
						'Unchanged_Cycling_Genes': unchanged_cycling_genes,
						'Unchanged_Noncycling_Genes': unchanged_noncycling_genes
					})
		
		# Create DataFrame
		results_df = pd.DataFrame(data_rows)
		
		# Calculate percentages
		results_df['% Changed'] = ((results_df['Cycling to Non-cycling'] + 
								  results_df['Non-cycling to Cycling']) / 
								 results_df['Total Genes'] * 100)
		
		# Define all possible combinations
		all_combinations = [
			('mother', 'Low'), ('mother', 'Medium'), ('mother', 'High'),
			('daughter', 'Low'), ('daughter', 'Medium'), ('daughter', 'High')
		]
		
		# Use specified selection or default to all
		selected_combinations = branch_expr_selection if branch_expr_selection else all_combinations
		
		# Determine grid layout based on number of selected plots
		n_plots = len(selected_combinations)
		if n_plots <= 3:
			n_rows, n_cols = 1, n_plots
		else:
			n_rows = (n_plots + 2) // 3  # Ceiling division
			n_cols = min(3, n_plots)
		
		# Create figure with appropriate grid
		fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
		
		# Handle single subplot case
		if n_plots == 1:
			axes = np.array([axes])
		
		# Flatten axes for easier indexing
		if n_plots > 1:
			axes = np.array(axes).flatten()
		
		# Set y-axis max for consistency - only based on changed categories
		y_max = 0
		for _, row in results_df.iterrows():
			y_max = max(y_max, row['Cycling to Non-cycling'], row['Non-cycling to Cycling'])
		
		# Add a small buffer to the y-axis max
		y_max = y_max * 1.2
		
		# Iterate through selected combinations
		for idx, (branch, expr) in enumerate(selected_combinations):
			ax = axes[idx]
			
			# Set title
			ax.set_title(f"{branch.title()} - {expr} Expression", fontsize=12)
			
			# Set axis labels
			ax.set_ylabel('Number of genes', fontsize=10)
			ax.set_xlabel('Replication Timing', fontsize=10)
			
			# Add placeholder if no data
			branch_expr_data = results_df[(results_df['Branch'] == branch) & 
										 (results_df['Expression'] == expr)]
			
			if len(branch_expr_data) == 0:
				ax.text(0.5, 0.5, "No data available", ha='center', va='center')
				continue
			
			# Plot bars for each replication timing
			x_offsets = [-0.25, 0, 0.25]  # Offset for positioning bars
			bar_width = 0.075
			
			for k, timing in enumerate(['early', 'mid', 'late']):
				timing_data = branch_expr_data[branch_expr_data['Replication'] == timing]
				
				if len(timing_data) == 0:
					continue
				
				# Get base replication color
				color = self.replication_colors[timing]

				# Extract counts
				cycle_to_noncycle = timing_data.iloc[0]['Cycling to Non-cycling']
				noncycle_to_cycle = timing_data.iloc[0]['Non-cycling to Cycling']
				unchanged_cycling = timing_data.iloc[0]['Unchanged Cycling']
				unchanged_noncycling = timing_data.iloc[0]['Unchanged Non-cycling']
				
				# Plot bars with offset for each replication timing
				x_pos = x_offsets[k]
				
				# First category: Cycling to Non-cycling
				bar1 = ax.bar(x_pos - bar_width/2, cycle_to_noncycle, 
							width=bar_width, color=color, 
							alpha=0.5,
							label=f"{timing.title()} (to non-cycling)" if idx == 0 else "")
				
				# Second category: Non-cycling to Cycling
				bar2 = ax.bar(x_pos + bar_width/2, noncycle_to_cycle, 
							width=bar_width, color=color,
							label=f"{timing.title()} (to cycling)" if idx == 0 else "")
				
				# Add count labels for changed categories
				if cycle_to_noncycle > 0:
					ax.text(x_pos - bar_width/2, cycle_to_noncycle + 0.5, 
						  str(int(cycle_to_noncycle)), ha='center', va='bottom', fontsize=8)
				
				if noncycle_to_cycle > 0:
					ax.text(x_pos + bar_width/2, noncycle_to_cycle + 0.5, 
						  str(int(noncycle_to_cycle)), ha='center', va='bottom', fontsize=8)
			
			# Set consistent y limit (only showing the changed categories properly)
			ax.set_ylim(0, y_max)

			ax.set_xticks(x_offsets)
			ax.set_xticklabels(['Early', 'Mid', 'Late'], fontsize=10)
			
			# Add grid
			ax.grid(axis='y', linestyle='-', lw=0.25, alpha=0.5)
		
		# Hide empty subplots if any
		for i in range(n_plots, len(axes)):
			axes[i].set_visible(False)

		plt.tight_layout()
		
		# Add overall title
		if title is None:
			title = f'{feature_title} PTR change following copy correction'

		if sup_y_adjust is None:
			sup_y_adjust = 1.05

		plt.suptitle(title, 
					 fontsize=18, fontweight='demi', y=sup_y_adjust)

		plt.subplots_adjust(bottom=0.15, hspace=0.45, wspace=0.25)

		return fig, results_df

	def create_violin_bar_legend(self, which='violin'):

		from src.plot_helpers import hide_spines

		fig, ax = plt.subplots(1, 1, figsize=(8.5, 0.5))

		# Create a custom legend
		legend_elements = []

		if which == 'violin':
			labels = ['Uncorrected', 'Copy corrected']
		elif which == 'bar':
			labels = ['To non-cycling', 'To cycling']
		
		for timing in ['early', 'mid', 'late']:
			color = self.replication_colors[timing]
			legend_elements.append(plt.Line2D([0], [0], color=color, lw=8, 
											label=f"{timing.title()}"))

		colors = ['#ddd', self.replication_colors['mid']]
		for i, label in enumerate(labels):
			legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=8, 
											label=f"{label}"))
		
		# Add legend to figure
		fig.legend(handles=legend_elements, loc='center',
				   ncol=5, fontsize=11)
		hide_spines(ax)

		return fig
