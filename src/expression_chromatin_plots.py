import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.plot_helpers import hide_spines
from src.plot_helpers import color_for_key
from src.config_utils import get_sample_indices
from src.orf_plotter import load_default_orf_plotter




# Note:   Deprecated, switching to ChromatinRNALocusPlotter.py
#         for RNA pileup plotting and single branch plotting
#         this code is useful for reference, as it contains
#         the old expression plotting code down the right side 
#         of the plot



class DeconvolutionChromatinExpressionPlotter:
	"""
	Class to plot the deconvolved chromatin context for a window.

	This class was originally used to plot the chromatin and expression profile
	for a gene. The gene expression is added as a third column that spans the entire vertical
	height. This plot has since been used to plot larger windows in which multiple genes
	are present and therefore expression must be plotted differently.
	"""

	def __init__(self, config1, figsize=(15, 5), plot_expression=False, title=None,
		branches_to_plot=["recovery", "mother", "daughter", "difference_mother_daughter"]):
		"""
		Initialize the plotter with configuration
		
		Parameters:
		-----------
		config1 : object
			Configuration object with methods:
			- get_Hpositions_for_phase(phase)
			- get_timepoints_for_phase(phase)
		branches_to_plot : list
			List of branch types to plot. Options:
			- "recovery": Recovery G1 phase
			- "mother": Mother G1 phase  
			- "daughter": Daughter G1 phase
			- "mean_mother_daughter": Average of mother and daughter
			- "difference_mother_daughter": Difference between daughter and mother
		"""
		self.config1 = config1
		self.title = title
		self.highlight_bins = []

		self.map_phase_name = {
			'R': "Recovery G1",
			'RG1': "Recovery G1",
			'CG1': "Mother G1",
			'meanG1': "Mean Mother-Daughter G1",
			'DG1': "Daughter G1",
			'postG1': "S/G2/M",
			'S': 'S',
			'G2M': 'G2/M',
		}
		
		self.branches_to_plot = branches_to_plot
		self.expression_F = None
		self.chromatin_F = None
		
		# Set row parameters
		self.num_rows = 11
		self.vmax = 40
		self.num_g1_rows = 6
		self.num_s_rows = 3
		self.num_g2m_rows = self.num_rows - self.num_g1_rows - self.num_s_rows
		self.plot_expression = plot_expression

		if plot_expression:
			expression_width = 1.5
			branch_spacing = 0.5
		else:
			expression_width = 0
			branch_spacing = 0.1

		number_of_branches = len(self.branches_to_plot)

		# Create the layout
		self.fig, self.chrom_axs, self.exp_axs, self.ann_axs, self.cell_cycle_axes = \
			create_chromatin_expression_layout(figsize=figsize, n_rows=self.num_rows,
				expression_width=expression_width, branch_spacing=branch_spacing,
				number_of_branches=number_of_branches)
			
		self.orf_plotter = load_default_orf_plotter()

		# Initialize the axes
		self._initialize_axes()
		
	def _initialize_axes(self):
		"""Setup the axes with proper formatting"""

		spine_border_width = 1.5

		def format_axes(ax):
			ax.set_xticks([])
			ax.set_yticks([])
			ax.spines['top'].set_linewidth(spine_border_width)
			ax.spines['bottom'].set_linewidth(spine_border_width)
			ax.spines['left'].set_linewidth(spine_border_width)
			ax.spines['right'].set_linewidth(spine_border_width)

		# Format all axes
		[format_axes(ax) for ax in np.array(self.chrom_axs).flatten()]
		[format_axes(ax) for ax in self.exp_axs]
		[format_axes(ax) for ax in self.ann_axs]
		[format_axes(ax) for ax in self.cell_cycle_axes]

		for chrom_branch_ax in self.chrom_axs:
			for i, ax in enumerate(chrom_branch_ax):
				internal_spine_width = 1
				top_spine = ax.spines['top']
				bottom_spine = ax.spines['bottom']
				internal_spines = []
				if i == 0:
					internal_spines = [bottom_spine]
				elif i == self.num_rows-1:
					internal_spines = [top_spine]
					top_spine.set_linewidth(internal_spine_width)
				else:
					internal_spines = [top_spine, bottom_spine]

				for spine in internal_spines:
					spine.set_linewidth(internal_spine_width)
					spine.set_color('#b0a996')

		# Hide expression spines for difference branches (they don't have expression data)
		for i, branch_type in enumerate(self.branches_to_plot):
			if branch_type == "difference_mother_daughter":
				hide_spines(self.exp_axs[i])


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

	def _get_phase_for_branch_type(self, branch_type):
		"""Get the phase string for cell cycle annotations based on branch type"""
		if branch_type == "recovery":
			return 'RG1'
		elif branch_type == "mother":
			return 'CG1'
		elif branch_type == "daughter":
			return 'DG1'
		elif branch_type == "mean_mother_daughter":
			return 'meanG1'
		elif branch_type == "difference_mother_daughter":
			return 'meanG1'  # Use mean for the difference display
		else:
			raise ValueError(f"Unknown branch type: {branch_type}")

	def _get_expression_phase_for_branch_type(self, branch_type):
		"""Get the phase string for expression plotting based on branch type"""
		if branch_type == "recovery":
			return 'RG1'
		elif branch_type == "mother":
			return 'CG1'
		elif branch_type == "daughter":
			return 'DG1'
		elif branch_type == "mean_mother_daughter":
			return 'CG1'  # Use CG1 as representative for mean
		elif branch_type == "difference_mother_daughter":
			return None  # No expression for difference
		else:
			raise ValueError(f"Unknown branch type: {branch_type}")

	def _plot_cell_cycle_annotations(self, axis_idx, branch_type):
		"""
		Plot cell cycle phase annotations with colored rectangles and text
		
		Parameters:
		-----------
		axis_idx : int
			Index of the cell cycle axis to plot on
		branch_type : str
			The branch type being plotted
		"""

		# Constants for positioning
		bar_width = 1.0
		bar_x = 0.0  # Position bar on right side
		text_x = 0.5 # Centered
		
		ax = self.cell_cycle_axes[axis_idx]
		g1_phase = self._get_phase_for_branch_type(branch_type)
		
		# Plot G1 rectangle and text
		g1_color = color_for_key(g1_phase)
		g1_rect = plt.Rectangle((bar_x, 0), bar_width, self.num_g1_rows, 
							  facecolor=g1_color, alpha=1)
		ax.add_patch(g1_rect)
		
		# Plot S rectangle and text
		s_color = color_for_key('S')
		s_rect = plt.Rectangle((bar_x, self.num_g1_rows), bar_width, self.num_s_rows, 
							   facecolor=s_color, alpha=1)
		ax.add_patch(s_rect)

		# Plot G2/M rectangle and text
		g2m_color = color_for_key('G2M')
		g2m_rect = plt.Rectangle((bar_x, self.num_g1_rows + self.num_s_rows), bar_width, self.num_g2m_rows, 
							   facecolor=g2m_color, alpha=1)
		ax.add_patch(g2m_rect)

		# Add vertical text labels
		# G1 phase text
		g1_center = self.num_g1_rows / 2
		ax.text(text_x, g1_center, self.map_phase_name[g1_phase], fontsize=18,
				rotation=270, va='center', ha='center', color='white',
				fontdict={'fontname': 'Open Sans'})
		
		# S phase text
		s_center = self.num_g1_rows + (self.num_s_rows / 2)
		ax.text(text_x, s_center, self.map_phase_name['S'], fontsize=18,
				rotation=270, va='center', ha='center', color='white',
				fontdict={'fontname': 'Open Sans'})

		# G2/M phase text
		g2m_center = self.num_g1_rows + self.num_s_rows + (self.num_g2m_rows / 2)
		ax.text(text_x, g2m_center, self.map_phase_name['G2M'], fontsize=18,
				rotation=270, va='center', ha='center', color='white',
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

	def plot_im(self, ax, img, cmap=plt.cm.magma_r, vmin=0, vmax=None):
		if vmax is None: vmax = self.vmax
		ax.imshow(img, aspect='auto', cmap=cmap, origin='lower', interpolation='none',
			vmin=vmin, vmax=vmax, extent=[self.span[0], self.span[1], 0, 260])

		from src.plot_helpers import plot_rect2

		x_bp_start, x_bp_end = self.span
		for color, bin_tuple in self.highlight_bins:
			x_bp = bin_tuple[0], bin_tuple[2]
			y_bp = bin_tuple[1]+5, bin_tuple[3]-5 # Inset slightly for visuals

			plot_rect2(ax, x_bp[0], y_bp[0], 
					   x_bp[1], y_bp[1], alpha=0.8,
				lw=0.75, fill=False, edgecolor=color)

	def _plot_mean_mother_daughter_chromatin(self, branch_idx):
		"""Plot the average of mother and daughter chromatin data"""
		
		# todo: CG1 == MG1 (as common G1, old nomenclature)
		dg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'DG1')
		mg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'CG1')
		
		# Plot G1 phase chromatin
		for row_idx in range(len(mg1_indices)):

			# Take the average of the MG1 and DG1 image
			dg1_idx = dg1_indices[row_idx]
			mg1_idx = mg1_indices[row_idx]
			dg1_img = self.chromatin_F[dg1_idx]
			mg1_img = self.chromatin_F[mg1_idx]
			mean_img = (dg1_img+mg1_img)/2.

			ax = self.chrom_axs[branch_idx][row_idx]
			self.plot_im(ax, mean_img)

		self._branch_plot_s_g2m(branch_idx)

	def _branch_plot_s_g2m(self, branch_idx):
		s_indices = get_sample_indices(self.config1, self.num_s_rows, 'S')
		g2m_indices = get_sample_indices(self.config1, self.num_g2m_rows, 'G2M')
		
		# Plot S phase chromatin
		for row_idx, idx in enumerate(s_indices):
			ax = self.chrom_axs[branch_idx][row_idx + self.num_g1_rows]
			self.plot_im(ax, self.chromatin_F[idx])

		# Plot G2M phase chromatin
		for row_idx, idx in enumerate(g2m_indices):
			ax = self.chrom_axs[branch_idx][row_idx + self.num_g1_rows + self.num_s_rows]
			self.plot_im(ax, self.chromatin_F[idx])


	def _plot_chromatin_branch(self, branch_idx, g1_phase):
		"""Plot chromatin data for a single branch"""
		
		# Get sampled indices for each phase
		g1_indices = get_sample_indices(self.config1, self.num_g1_rows, g1_phase)
	
		# Plot G1 phase chromatin
		for row_idx, idx in enumerate(g1_indices):
			ax = self.chrom_axs[branch_idx][row_idx]
			self.plot_im(ax, self.chromatin_F[idx])

		self._branch_plot_s_g2m(branch_idx)


	def _plot_difference_mother_daughter_chromatin(self, branch_idx):
		"""Plot difference between daughter and mother chromatin data"""
		
		# Get sampled indices for each phase
		dg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'DG1')
		cg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'CG1')

		# Plot G1 phase chromatin difference
		vmax = 10

		for row_idx in range(len(dg1_indices)):
			ax = self.chrom_axs[branch_idx][row_idx]

			dg1_dat = self.chromatin_F[dg1_indices[row_idx]]
			cg1_dat = self.chromatin_F[cg1_indices[row_idx]]
			diff = dg1_dat-cg1_dat
			self.plot_im(ax, diff, vmin=-vmax, vmax=vmax, cmap='RdBu_r')


	def _plot_chromatin_for_branch_type(self, branch_idx, branch_type):
		"""Plot chromatin data based on branch type"""
		if branch_type == "recovery":
			self._plot_chromatin_branch(branch_idx, 'RG1')
		elif branch_type == "mother":
			self._plot_chromatin_branch(branch_idx, 'CG1')
		elif branch_type == "daughter":
			self._plot_chromatin_branch(branch_idx, 'DG1')
		elif branch_type == "mean_mother_daughter":
			self._plot_mean_mother_daughter_chromatin(branch_idx)
		elif branch_type == "difference_mother_daughter":
			self._plot_difference_mother_daughter_chromatin(branch_idx)
		else:
			raise ValueError(f"Unknown branch type: {branch_type}")

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
		self.orf_plotter.set_chrom_span(chrom, span)

	def _plot_annotations(self):
		"""Plot gene annotations on all annotation axes"""
		for ax in self.ann_axs:
			self.orf_plotter.plot_orf_annotations(ax)

	def plot(self):
		"""Plot the expression data for all branches"""
		
		# Validate that required data is available
		if self.chromatin_F is None:
			raise ValueError("Chromatin data not set. Call set_chromatin_data first.")
		
		if self.plot_expression and self.expression_F is None:
			raise ValueError("Expression data not set but plot_expression=True. Call set_expression_data first.")
		
		# Clear the axes for redraw
		self.clear_axes()

		# Plot gene annotations
		self._plot_annotations()

		# Plot each branch based on branches_to_plot
		for i, branch_type in enumerate(self.branches_to_plot):
			# Plot cell cycle annotations
			self._plot_cell_cycle_annotations(i, branch_type)
			
			# Plot expression if enabled and available for this branch type
			if self.plot_expression and self.expression_F is not None:
				expression_phase = self._get_expression_phase_for_branch_type(branch_type)
				if expression_phase is not None:
					self._plot_expression_branch(i, expression_phase)
			
			# Plot chromatin data
			self._plot_chromatin_for_branch_type(i, branch_type)

		# Set title
		title = self.title
		if title is None:
			title = f"chr{self.chrom}, {self.span[0]}-{self.span[1]}"
		plt.suptitle(title, fontsize=32, y=1, fontweight='demi')

		# For the last row of the chromatin axs, show ticks for positions 500 kb apart
		minor_xticks = np.arange(self.span[0], self.span[1]+200, 200)
		major_xticks = np.arange(self.span[0], self.span[1]+1000, 1000)

		# For each branch (set of chromatin axes), add ticks to the bottom row
		for branch_axs in self.chrom_axs:
			ax = branch_axs[-1]

			# Set the tick mark locations
			ax.set_xticks(minor_xticks, minor=True)
			ax.set_xticks(major_xticks, minor=False)

			# Formatting
			ax.tick_params(axis='x', which='major', length=5, width=1.25, labelbottom=False)
			ax.tick_params(axis='x', which='minor', length=2, width=1, labelbottom=False)
			ax.set_xlim(*self.span)

		return self.fig


# Keep the existing layout function unchanged
def create_chromatin_expression_layout(
	n_rows=7,  # Number of rows in chromatin data (excluding annotation)
	figsize=(10, 5),  # Figure size
	chromatin_width_ratios=[1, 1, 1, 1, 1],  # Width ratios for chromatin columns, total column width = 4
	expression_width=1.5,  # Width of expression plot relative to chromatin (1/4)
	cell_cycle_annotations_width=0.25,  # Width of cell cycle annotations plot relative to chromatin (1/4)
	branch_spacing=0.5,  # Spacing between branches
	top_margin=0.95,  # Top margin for titles
	bottom_margin=0.1,  # Bottom margin
	height_ratios=None,  # Optional custom height ratios for rows
	annotation_height=1.2,  # Height of annotation row relative to data rows
	annotation_spacing=0.2,  # Height of spacing between annotation and data rows
	number_of_branches = 4 # Number of branches
):
	"""
	Creates a layout for chromatin and expression data visualization with gene annotations
	and cell cycle indicators.
	"""
	# Calculate the number of columns needed
	n_branches = number_of_branches  # Recovery, Mother, Daughter
	cols_per_branch = 3  # Cell cycle, chromatin, and expression
	
	# Create figure
	fig = plt.figure(figsize=figsize)
	
	# Calculate width ratios for all columns including spacing
	width_ratios = []
	for branch in range(n_branches):
		# Add cell cycle column
		width_ratios.append(cell_cycle_annotations_width)
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


# Keep the existing function unchanged
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
