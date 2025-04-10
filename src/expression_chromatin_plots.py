
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.plot_helpers import hide_spines
from src.plot_helpers import color_for_key
from src.config_utils import get_sample_indices
from src.orf_plotter import load_default_orf_plotter


class DeconvolutionChromatinExpressionPlotter:
	def __init__(self, config1):
		"""
		Initialize the plotter with configuration
		
		Parameters:
		-----------
		config1 : object
			Configuration object with methods:
			- get_Hpositions_for_phase(phase)
			- get_timepoints_for_phase(phase)
		"""
		self.config1 = config1

		self.map_phase_name = {
			'R': "Recovery G1",
			'RG1': "Recovery G1",
			'CG1': "Mother G1",
			'DG1': "Daughter G1",
			'postG1': "S/G2/M",
		}

		self.expression_F = None
		
		# Set row parameters
		self.num_rows = 7
		self.vmax = 40
		self.num_g1_rows = 4
		self.num_pg1_rows = self.num_rows - self.num_g1_rows

		# # Save the current interactive state
		# was_interactive = plt.isinteractive()
		
		# # Temporarily turn off interactive mode
		# if was_interactive:
		# 	plt.ioff()

		# Create the layout
		self.fig, self.chrom_axs, self.exp_axs, self.ann_axs, self.cell_cycle_axes = \
			create_chromatin_expression_layout(figsize=(15, 5), n_rows=self.num_rows)
		
		# # Restore previous interactive state
		# if was_interactive:
		# 	plt.ion()
		
		# # Make sure the figure doesn't display yet
		# plt.close(self.fig)
			
		self.orf_plotter = load_default_orf_plotter()

		# Initialize the axes
		self._initialize_axes()
		
	def _initialize_axes(self):
		"""Setup the axes with proper formatting"""
		def hide_ticks(ax):
			ax.set_xticks([])
			ax.set_yticks([])
		
		# Hide ticks on all axes
		[hide_ticks(ax) for ax in np.array(self.chrom_axs).flatten()]
		[hide_ticks(ax) for ax in self.exp_axs]
		[hide_ticks(ax) for ax in self.ann_axs]
		[hide_ticks(ax) for ax in self.cell_cycle_axes]
		
		# Hide spines on specific axes
		hide_spines(self.exp_axs[3])
		[hide_spines(ax) for ax in self.cell_cycle_axes]

	def clear_axes(self):
		"""Clear all axes in the plot"""
		# Clear chromatin axes
		for branch_axes in self.chrom_axs:
			for ax in branch_axes:
				plt.sca(ax)
				plt.cla()
		
		# Clear expression axes
		for ax in self.exp_axs:
			plt.sca(ax)
			plt.cla()
			
		# Clear annotation axes
		for ax in self.ann_axs:
			plt.sca(ax)
			plt.cla()
			
		# Clear cell cycle axes
		for ax in self.cell_cycle_axes:
			plt.sca(ax)
			plt.cla()

		self._initialize_axes()
	
	def _get_tx_indices_tps_phase(self, phase):
		"""Get indices, timepoints and tx data for a given phase"""
		indices = self.config1.get_Hpositions_for_phase(phase)
		tps = self.config1.get_timepoints_for_phase(phase)
		tx = self.expression_F[indices]
		return indices, tps, tx
	
	def _plot_expression_branch(self, ax_idx, phase):
		"""Plot expression data for a single branch"""
		
		# Round up to the nearest 10 if necessary
		round_num = 2
		max_xlim = self.expression_F.max() * 1.15
		max_xlim = max(round_num, (max_xlim//round_num+1)*round_num)

		ax = self.exp_axs[ax_idx]

		# Plot G1 phase
		color = color_for_key(phase)
		indices, g1_tps, g1_tx = self._get_tx_indices_tps_phase(phase)
		y_values = np.linspace(0, self.num_g1_rows, len(g1_tx))
		ax.fill_betweenx(y_values, 0, g1_tx, color=color)
		
		# Plot post-G1 phase
		color = color_for_key('postG1')
		indices, pg1_tps, pg1_tx = self._get_tx_indices_tps_phase('postG1')
		y_values = np.linspace(self.num_g1_rows, self.num_rows, len(pg1_tx))
		ax.fill_betweenx(y_values, 0, pg1_tx, color=color)
		
		ax.set_xlim(0, max_xlim)
		ax.set_ylim(self.num_rows, 0)

		# Round to the nearest
		ax.set_xticks(np.arange(0, max_xlim+2, 2))

	def _plot_cell_cycle_annotations(self, axis_idx, g1_phase):
		"""
		Plot cell cycle phase annotations with colored rectangles and text
		
		Parameters:
		-----------
		axis_idx : int
			Index of the cell cycle axis to plot on
		g1_phase : str
			The G1 phase to plot ('RG1', 'CG1', or 'DG1')
		"""

		# Constants for positioning
		bar_width = 0.3
		bar_x = 0.7  # Position bar on right side
		text_x = 0.4  # Position text on left side
		
		ax = self.cell_cycle_axes[axis_idx]
		
		# Plot G1 rectangle and text
		g1_color = color_for_key(g1_phase)
		g1_rect = plt.Rectangle((bar_x, 0), bar_width, self.num_g1_rows, 
							  facecolor=g1_color, alpha=1)
		ax.add_patch(g1_rect)
		
		# Plot postG1 rectangle and text
		pg1_color = color_for_key('postG1')
		pg1_rect = plt.Rectangle((bar_x, self.num_g1_rows), bar_width, self.num_pg1_rows, 
							   facecolor=pg1_color, alpha=1)
		ax.add_patch(pg1_rect)

		# Add vertical text labels
		# G1 phase text
		g1_center = self.num_g1_rows / 2
		ax.text(text_x, g1_center, self.map_phase_name[g1_phase],
				rotation=270, va='center', ha='center', color='black',
				fontdict={'fontname': 'Open Sans'})
		
		# PostG1 phase text
		pg1_center = self.num_g1_rows + (self.num_pg1_rows / 2)
		ax.text(text_x, pg1_center, self.map_phase_name['postG1'],
				rotation=270, va='center', ha='center', color='black',
				fontdict={'fontname': 'Open Sans'})
		
		# Set axis limits
		ax.set_xlim(0, 1)
		ax.set_ylim(self.num_rows, 0)

	
	def set_expression_data(self, F):
		"""
		Set the expression data to be plotted
		
		Parameters:
		-----------
		F : numpy.ndarray
			Expression data array
		"""
		self.expression_F = F

		
	def set_chromatin_data(self, F):
		"""
		Set the chromatin data to be plotted
		
		Parameters:
		-----------
		F : numpy.ndarray
			Chromatin data array
		"""
		self.chromatin_F = F

	def _plot_chromatin_branch(self, branch_idx, g1_phase):
		"""Plot chromatin data for a single branch"""
		if self.chromatin_F is None:
			raise ValueError("Chromatin data not set. Call set_chromatin_data first.")
			
		# Get sampled indices for each phase
		g1_indices = get_sample_indices(self.config1, self.num_g1_rows, g1_phase)
		pg1_indices = get_sample_indices(self.config1, self.num_pg1_rows, 'postG1')
		
		# Plot G1 phase chromatin
		for row_idx, idx in enumerate(g1_indices):
			ax = self.chrom_axs[branch_idx][row_idx]
			ax.imshow(self.chromatin_F[idx], aspect='auto', cmap='magma_r', origin='lower', 
			vmin=0, vmax=self.vmax)
			
		# Plot postG1 phase chromatin
		for row_idx, idx in enumerate(pg1_indices):
			ax = self.chrom_axs[branch_idx][row_idx + self.num_g1_rows]
			ax.imshow(self.chromatin_F[idx], aspect='auto', cmap='magma_r', origin='lower', 
			vmin=0, vmax=self.vmax)



	def _plot_chromatin_branch_difference(self, branch_idx):
		"""Plot chromatin data for a single branch"""
		if self.chromatin_F is None:
			raise ValueError("Chromatin data not set. Call set_chromatin_data first.")
			
		# Get sampled indices for each phase
		dg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'DG1')
		cg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'CG1')
		pg1_indices = get_sample_indices(self.config1, self.num_pg1_rows, 'postG1')
		
		# Plot G1 phase chromatin
		vmax = 10

		for row_idx in range(len(dg1_indices)):
			ax = self.chrom_axs[branch_idx][row_idx]

			dg1_dat = self.chromatin_F[dg1_indices[row_idx]]
			cg1_dat = self.chromatin_F[cg1_indices[row_idx]]
			diff = dg1_dat-cg1_dat

			ax.imshow(diff, aspect='auto', cmap='RdBu_r', origin='lower', 
			vmin=-vmax, vmax=vmax)


	def set_chrom_span(self, chrom, span):
		"""
		Set chromosome and span for gene annotations
		
		Parameters:
		-----------
		chrom : str
			Chromosome identifier
		span : tuple
			Tuple of (start, end) positions
		"""
		self.chrom = chrom
		self.span = span
		self.orf_plotter.set_span_chrom(span, chrom)

	def _plot_annotations(self):
		"""Plot gene annotations on all annotation axes"""
		for ax in self.ann_axs:
			self.orf_plotter.plot_orf_annotations(ax)

	def plot(self, title=None):
		"""Plot the expression data for all branches"""
		
		# Clear the axes for redraw
		self.clear_axes()

		# Plot gene annotations
		self._plot_annotations()

		# Plot cell cycle annotations
		self._plot_cell_cycle_annotations(0, 'RG1')
		self._plot_cell_cycle_annotations(1, 'CG1')
		self._plot_cell_cycle_annotations(2, 'DG1')

		# Plot each branch
		if self.expression_F is not None:
			self._plot_expression_branch(0, 'RG1')
			self._plot_expression_branch(1, 'CG1')
			self._plot_expression_branch(2, 'DG1')

		# Plot chromatin for each branch
		self._plot_chromatin_branch(0, 'RG1')
		self._plot_chromatin_branch(1, 'CG1')
		self._plot_chromatin_branch(2, 'DG1')

		self._plot_chromatin_branch_difference(3)

		# Set title
		if title is None:
			title = f"chr{self.chrom}, {self.span[0]}-{self.span[1]}"
		plt.suptitle(title)
		
		return self.fig


def create_chromatin_expression_layout(
	n_rows=7,  # Number of rows in chromatin data (excluding annotation)
	figsize=(10, 5),  # Figure size
	chromatin_width_ratios=[1, 1, 1, 1, 1],  # Width ratios for chromatin columns, total column width = 4
	expression_width=1.5,  # Width of expression plot relative to chromatin (1/4)
	cell_cycle_width=0.75,  # Width of cell cycle plot relative to chromatin (1/4)
	branch_spacing=0.5,  # Spacing between branches
	top_margin=0.90,  # Top margin for titles
	bottom_margin=0.05,  # Bottom margin
	height_ratios=None,  # Optional custom height ratios for rows
	annotation_height=0.75,  # Height of annotation row relative to data rows
	annotation_spacing=0.15  # Height of spacing between annotation and data rows
):
	"""
	Creates a layout for chromatin and expression data visualization with gene annotations
	and cell cycle indicators.
	"""
	# Calculate the number of columns needed
	n_branches = 4  # Recovery, Mother, Daughter
	cols_per_branch = 3  # Cell cycle, chromatin, and expression
	
	# Create figure
	fig = plt.figure(figsize=figsize)
	
	# Calculate width ratios for all columns including spacing
	width_ratios = []
	for branch in range(n_branches):
		# Add cell cycle column
		width_ratios.append(cell_cycle_width)
		# Add chromatin columns
		width_ratios.extend(chromatin_width_ratios)
		# Add expression column
		width_ratios.append(expression_width)
		# Add spacing (except after last branch)
		if branch < n_branches - 1:
			width_ratios.append(branch_spacing)
	
	# Create height ratios including annotation row and spacing row
	if height_ratios is None:
		height_ratios = [annotation_height, annotation_spacing] + [1] * n_rows
	else:
		height_ratios = [annotation_height, annotation_spacing] + height_ratios
	
	# Create GridSpec
	gs = gridspec.GridSpec(
		n_rows + 2,  # Add 2 for annotation row and spacing
		len(width_ratios),
		width_ratios=width_ratios,
		height_ratios=height_ratios,
		hspace=0,
		wspace=0
	)
	
	# Create axes for all plot types
	chromatin_axes = [[] for _ in range(n_branches)]
	expression_axes = []
	annotation_axes = []
	cell_cycle_axes = []
	
	# Calculate column indices for each branch
	col_idx = 0
	for branch in range(n_branches):
		# Cell cycle column index
		cycle_idx = col_idx
		# Chromatin start column index
		chrom_idx = col_idx + 1
		# Expression column index
		expr_idx = chrom_idx + len(chromatin_width_ratios)
		
		# Create annotation axis (spans only chromatin)
		ax = fig.add_subplot(gs[0, chrom_idx:expr_idx])
		annotation_axes.append(ax)
		
		# Create cell cycle axis (skipping annotation and spacing rows)
		ax = fig.add_subplot(gs[2:, cycle_idx])
		cell_cycle_axes.append(ax)
		
		# Create chromatin axes (starting after annotation and spacing rows)
		for row in range(n_rows):
			ax = fig.add_subplot(gs[row + 2, chrom_idx:expr_idx])
			chromatin_axes[branch].append(ax)
		
		# Create expression axis (skipping annotation and spacing rows)
		ax = fig.add_subplot(gs[2:, expr_idx])
		expression_axes.append(ax)
		
		# Move to next branch (including spacing)
		col_idx = expr_idx + 2 if branch < n_branches - 1 else expr_idx + 1
	
	# Adjust layout
	plt.subplots_adjust(
		top=top_margin,
		bottom=bottom_margin
	)
	
	return fig, chromatin_axes, expression_axes, annotation_axes, cell_cycle_axes



def draw_phase_label_annotations(ax, config=None, 
		phases = ['CG1', 'S', 'G2M'], 
		phase_names = ['CG1', 'S', 'GSM'],
		flip=False, annotations_x=0, offset=False):

	from src.plot_helpers import color_for_key

	tp_set = []
	for phase in phases:
		tps = config.get_timepoints_for_phase(phase)
		tp_set.append(tps)

	if offset:
		offset_by = -tp_set[0][0]

	last_tp = None
	for i in range(len(tp_set)):
		tps = tp_set[i]

		if len(tps) == 0: continue

		if offset: tps = tps + offset_by

		phase = phases[i]
		phase_name = phase_names[i]

		if last_tp is None:
			last_tp = tps[0]

		tp_start, tp_end = last_tp, tps[-1]

		last_tp = tps[-1]

		tp_mid = (tp_start + tp_end)/2

		xs = [annotations_x, annotations_x]
		ys = [tp_start, tp_end]

		text_x = annotations_x
		text_y = tp_mid
		rotation = 90

		if flip:
			xs, ys = ys, xs
			text_x, text_y = text_y, text_x
			rotation = 0

		ax.plot(xs, ys, c=color_for_key(phase), lw=20, solid_capstyle='butt')

		ax.text(text_x, text_y, phase_name, va='center', ha='center', fontsize=10,
			color='white', rotation=rotation)

import pandas as pd
import numpy as np

def categorize_cell_cycle_genes(expression_df, chromatin_df):
	"""
	Categorize genes based on their cell cycle activity in expression and chromatin data.
	
	Parameters:
	-----------
	expression_df : pandas.DataFrame
		DataFrame containing expression data with 'bin' column indicating cell cycle phase
	chromatin_df : pandas.DataFrame
		DataFrame containing chromatin data with 'bin' column indicating cell cycle phase
		
	Returns:
	--------
	dict
		Dictionary containing categorized ORF lists and a summary DataFrame
	"""
	# Define cell cycle categories
	cell_cycle_bins = ['MG1', 'DG1', 'postG1']
	
	# Create mask for cell cycle genes in each dataset
	expr_cell_cycle_mask = expression_df['bin'].isin(cell_cycle_bins)
	chrom_cell_cycle_mask = chromatin_df['bin'].isin(cell_cycle_bins)
	
	# Get unique ORFs from both datasets
	expr_orfs = set(expression_df.index)
	chrom_orfs = set(chromatin_df.index)
	all_orfs = expr_orfs.union(chrom_orfs)
	
	# Initialize results dictionary
	results = {
		'expression_only': [],
		'chromatin_only': [],
		'both': [],
		'either': [],
		'neither': [],
		'details': {}
	}
	
	# Create a dictionary to store detailed information for each ORF
	details = {}
	
	# Analyze each ORF
	for orf in all_orfs:
		expr_status = "Not found"
		chrom_status = "Not found"
		
		# Check if ORF is in expression dataset
		if orf in expr_orfs:
			if expr_cell_cycle_mask.loc[orf]:
				expr_status = expression_df.loc[orf, 'bin']
			else:
				expr_status = "Not cell cycle"
		
		# Check if ORF is in chromatin dataset
		if orf in chrom_orfs:
			if chrom_cell_cycle_mask.loc[orf]:
				chrom_status = chromatin_df.loc[orf, 'bin']
			else:
				chrom_status = "Not cell cycle"
		
		# Store detailed information
		details[orf] = {
			'expression': expr_status,
			'chromatin': chrom_status
		}
		
		# Categorize the ORF
		is_expr_cycling = expr_status in cell_cycle_bins
		is_chrom_cycling = chrom_status in cell_cycle_bins
		
		if is_expr_cycling and is_chrom_cycling:
			results['both'].append(orf)
		elif is_expr_cycling:
			results['expression_only'].append(orf)
		elif is_chrom_cycling:
			results['chromatin_only'].append(orf)
		else:
			results['neither'].append(orf)
		
		if is_expr_cycling or is_chrom_cycling:
			results['either'].append(orf)
	
	# Convert details to DataFrame for easy analysis
	results['details'] = pd.DataFrame(details).T
	results['details'].index.name = 'ORF'
	
	# Generate summary statistics
	results['summary'] = {
		'total_orfs': len(all_orfs),
		'expression_only': len(results['expression_only']),
		'chromatin_only': len(results['chromatin_only']),
		'both': len(results['both']),
		'either': len(results['either']),
		'neither': len(results['neither'])
	}
	
	return results


# -------------- Intersection of chromatin and expression data plots


def get_phase_specific_orfs(expression_df, chromatin_df, phase, data_source='either'):
	"""
	Get ORFs for a specific cell cycle phase from the specified data source.
	
	Parameters:
	-----------
	expression_df : pandas.DataFrame
		DataFrame containing expression data with 'bin' column
	chromatin_df : pandas.DataFrame
		DataFrame containing chromatin data with 'bin' column
	phase : str or list
		Cell cycle phase(s) to select ('MG1', 'DG1', 'postG1' or a list of these)
	data_source : str
		Which dataset to use: 'expression', 'chromatin', 'both', or 'either'
		
	Returns:
	--------
	list
		List of ORFs matching the criteria
	"""
	if isinstance(phase, str):
		phase = [phase]
	
	# Create masks for the specified phase
	expr_mask = expression_df['bin'].isin(phase)
	chrom_mask = chromatin_df['bin'].isin(phase)
	
	# Get ORFs based on data source
	if data_source == 'expression':
		return list(expression_df[expr_mask].index)
	elif data_source == 'chromatin':
		return list(chromatin_df[chrom_mask].index)
	elif data_source == 'both':
		expr_orfs = set(expression_df[expr_mask].index)
		chrom_orfs = set(chromatin_df[chrom_mask].index)
		return list(expr_orfs.intersection(chrom_orfs))
	elif data_source == 'either':
		expr_orfs = set(expression_df[expr_mask].index)
		chrom_orfs = set(chromatin_df[chrom_mask].index)
		return list(expr_orfs.union(chrom_orfs))
	else:
		raise ValueError("data_source must be one of: 'expression', 'chromatin', 'both', 'either'")

def generate_summary_report(results):
	"""
	Generate a summary report of cell cycle gene categorization.
	
	Parameters:
	-----------
	results : dict
		Results dictionary from categorize_cell_cycle_genes
		
	Returns:
	--------
	str
		Formatted summary report
	"""
	summary = results['summary']
	details = results['details']
	
	# Count phase-specific genes
	phase_counts = {
		'MG1': {
			'expression': sum(details['expression'] == 'MG1'),
			'chromatin': sum(details['chromatin'] == 'MG1'),
			'both': sum((details['expression'] == 'MG1') & (details['chromatin'] == 'MG1'))
		},
		'DG1': {
			'expression': sum(details['expression'] == 'DG1'),
			'chromatin': sum(details['chromatin'] == 'DG1'),
			'both': sum((details['expression'] == 'DG1') & (details['chromatin'] == 'DG1'))
		},
		'postG1': {
			'expression': sum(details['expression'] == 'postG1'),
			'chromatin': sum(details['chromatin'] == 'postG1'),
			'both': sum((details['expression'] == 'postG1') & (details['chromatin'] == 'postG1'))
		}
	}
	
	
	report = "Cell Cycle Gene Analysis Summary\n"
	report += "================================\n\n"
	report += f"Total ORFs analyzed: {summary['total_orfs']}\n\n"
	report += "Cell Cycle Categorization:\n"
	report += f"- Expression only: {summary['expression_only']} ORFs\n"
	report += f"- Chromatin only: {summary['chromatin_only']} ORFs\n"
	report += f"- Both: {summary['both']} ORFs\n"
	report += f"- Either: {summary['either']} ORFs\n"
	report += f"- Neither: {summary['neither']} ORFs\n\n"
	
	report += "Phase-Specific Counts:\n"
	for phase in ['MG1', 'DG1', 'postG1']:
		report += f"- {phase}:\n"
		report += f"  - Expression: {phase_counts[phase]['expression']} ORFs\n"
		report += f"  - Chromatin: {phase_counts[phase]['chromatin']} ORFs\n"
		report += f"  - Both: {phase_counts[phase]['both']} ORFs\n"
	return report


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_cell_cycle_comparison(results, figsize=(15, 12)):
	"""
	Create a visualization of cell cycle gene categorization results
	structured according to the summary grouping.
	
	Parameters:
	-----------
	results : dict
		Results dictionary from categorize_cell_cycle_genes
	figsize : tuple
		Figure size (width, height) in inches
		
	Returns:
	--------
	matplotlib.figure.Figure
		The generated figure
	"""
	details = results['details']
	
	# Set up the figure with subplots
	fig = plt.figure(figsize=figsize)
	
	# Define a grid specification for better layout control
	gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])
	
	# 1. Overall categorization counts (top row)
	ax1 = fig.add_subplot(gs[0, :])
	categories = ['Expression only', 'Chromatin only', 'Both', 'Either', 'Neither']
	counts = [
		results['summary']['expression_only'],
		results['summary']['chromatin_only'],
		results['summary']['both'],
		results['summary']['either'],
		results['summary']['neither']
	]
	
	colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#d3d3d3']
	bars = ax1.bar(categories, counts, color=colors)
	ax1.set_title('Cell Cycle Gene Classification', fontsize=14, fontweight='bold')
	ax1.set_ylabel('Number of Genes', fontsize=12)
	ax1.tick_params(axis='x', labelsize=11)
	ax1.tick_params(axis='y', labelsize=10)
	
	# Add count labels on top of bars
	for bar in bars:
		height = bar.get_height()
		ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
				f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
	
	# 2. Phase-specific counts by dataset (middle row)
	ax2 = fig.add_subplot(gs[1, :])
	
	# Prepare data for phase-specific counts
	phases = ['MG1', 'DG1', 'postG1']
	phase_counts = {
		'Expression': [
			sum(details['expression'] == 'MG1'),
			sum(details['expression'] == 'DG1'),
			sum(details['expression'] == 'postG1')
		],
		'Chromatin': [
			sum(details['chromatin'] == 'MG1'),
			sum(details['chromatin'] == 'DG1'),
			sum(details['chromatin'] == 'postG1')
		],
		'Both': []
	}
	
	# Calculate "Both" counts (genes with the same phase in both datasets)
	for phase in phases:
		both_count = sum((details['expression'] == phase) & (details['chromatin'] == phase))
		phase_counts['Both'].append(both_count)
	
	# Set up bar positions
	x = np.arange(len(phases))
	width = 0.25
	
	# Plot bars for each dataset
	bars1 = ax2.bar(x - width, phase_counts['Expression'], width, label='Expression', color='#ff7f0e')
	bars2 = ax2.bar(x, phase_counts['Chromatin'], width, label='Chromatin', color='#1f77b4')
	bars3 = ax2.bar(x + width, phase_counts['Both'], width, label='Both', color='#2ca02c')
	
	# Add count labels
	for bars in [bars1, bars2, bars3]:
		for bar in bars:
			height = bar.get_height()
			ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
					f'{int(height)}', ha='center', va='bottom', fontsize=10)
	
	ax2.set_title('Phase-Specific Gene Counts', fontsize=14, fontweight='bold')
	ax2.set_ylabel('Number of Genes', fontsize=12)
	ax2.set_xticks(x)
	ax2.set_xticklabels(phases, fontsize=11)
	ax2.legend(fontsize=11)
	
	# 3. Phase concordance matrix (bottom row) - using matplotlib instead of seaborn
	ax3 = fig.add_subplot(gs[2, :])
	
	# Create a cross-tabulation of phases - excluding "Not found"
	valid_phases = ['MG1', 'DG1', 'postG1', 'Not cell cycle']
	crosstab = pd.crosstab(
		details['expression'], 
		details['chromatin'],
		rownames=['Expression'], 
		colnames=['Chromatin']
	)
	
	# Ensure all valid phases are present and remove 'Not found'
	for phase in valid_phases:
		if phase not in crosstab.index:
			crosstab.loc[phase] = 0
		if phase not in crosstab.columns:
			crosstab[phase] = 0
	
	# Filter to keep only valid phases
	crosstab = crosstab.loc[valid_phases, valid_phases]
	
	# Create the heatmap using matplotlib's pcolormesh
	im = ax3.pcolormesh(crosstab.values, cmap='YlGnBu', vmax=200)
	
	# Set tick positions and labels
	ax3.set_xticks(np.arange(len(crosstab.columns)) + 0.5)
	ax3.set_yticks(np.arange(len(crosstab.index)) + 0.5)
	ax3.set_xticklabels(crosstab.columns, fontsize=11)
	ax3.set_yticklabels(crosstab.index, fontsize=11)
	
	# Put the text labels in the cells
	for i in range(len(crosstab.index)):
		for j in range(len(crosstab.columns)):
			val = crosstab.iloc[i, j]
			# Only show text for non-zero values
			if val > 0:
				# Decide on text color based on cell color
				cell_value = crosstab.iloc[i, j]
				max_value = crosstab.values.max()
				text_color = 'white' if cell_value > 150 else 'black'
				ax3.text(j + 0.5, i + 0.5, str(int(val)), 
						 ha='center', va='center', color=text_color, fontsize=12, 
						 fontweight='bold')
	
	# Add a colorbar
	cbar = plt.colorbar(im, ax=ax3)
	cbar.set_label('Number of Genes', fontsize=11)
	
	ax3.set_title('Phase Concordance Between Datasets', fontsize=14, fontweight='bold')
	
	# Add labels for the axes
	ax3.set_xlabel('Chromatin Data', fontsize=12)
	ax3.set_ylabel('Expression Data', fontsize=12)
	
	# Adjust layout
	plt.tight_layout()
	fig.subplots_adjust(hspace=0.3)
	
	return fig
