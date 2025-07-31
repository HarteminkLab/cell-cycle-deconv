
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.plot_helpers import hide_spines
from src.plot_helpers import color_for_key
from src.config_utils import get_sample_indices
from src.orf_plotter import load_default_orf_plotter
from src.rna_pileup_plotter import RNASeqPileupPlotter


class SingleBranchChromatinPlotter:
	"""
	Class to plot the chromatin context for a single branch.
	
	This class focuses on plotting chromatin data for one specific branch type
	(recovery, mother, daughter, mean, or difference) with gene annotations,
	RNA pileup, and cell cycle annotations.
	"""

	def __init__(self, outdir, config1, branch_type="mother", figsize=(6, 5), title=None,
		rna_plotter=None):
		"""
		Initialize the plotter with configuration for a single branch
		
		Parameters:
		-----------
		config1 : object
			Configuration object with methods:
			- get_Hpositions_for_phase(phase)
			- get_timepoints_for_phase(phase)
		branch_type : str
			Branch type to plot. Options:
			- "recovery": Recovery G1 phase
			- "mother": Mother G1 phase  
			- "daughter": Daughter G1 phase
			- "mean_mother_daughter": Average of mother and daughter
			- "difference_mother_daughter": Difference between daughter and mother
		figsize : tuple
			Figure size (width, height)
		title : str, optional
			Plot title
		"""
		self.config1 = config1
		self.title = title
		self.branch_type = branch_type
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
		
		self.chromatin_F = None
		
		# Set row parameters
		self.num_rows = 11
		self.vmax = 40
		self.num_g1_rows = 6
		self.num_s_rows = 3
		self.num_g2m_rows = self.num_rows - self.num_g1_rows - self.num_s_rows

		# Create the layout using our new single-branch layout function
		self.fig, self.chromatin_axes, self.annotation_axis, self.rna_pileup_axis, self.cell_cycle_axis = \
			create_single_branch_chromatin_layout(
				figsize=figsize, 
				n_rows=self.num_rows
			)
			
		self.orf_plotter = load_default_orf_plotter()

		if rna_plotter is None:
			self.rna_plotter = RNASeqPileupPlotter(outdir)
		else:
			self.rna_plotter = rna_plotter

		# Initialize the axes
		self._initialize_axes()
		
	def _initialize_axes(self):
		"""Setup the axes with proper formatting"""

		spine_border_width = 1.5
		internal_spine_width = 1

		def format_axes(ax):
			ax.set_xticks([])
			ax.set_yticks([])
			ax.spines['top'].set_linewidth(spine_border_width)
			ax.spines['bottom'].set_linewidth(spine_border_width)
			ax.spines['left'].set_linewidth(spine_border_width)
			ax.spines['right'].set_linewidth(spine_border_width)

		# Format all axes
		for ax in self.chromatin_axes:
			format_axes(ax)
		format_axes(self.annotation_axis)
		format_axes(self.rna_pileup_axis)
		format_axes(self.cell_cycle_axis)

		# Handle internal spines for chromatin axes
		for i, ax in enumerate(self.chromatin_axes):
			top_spine = ax.spines['top']
			bottom_spine = ax.spines['bottom']
			internal_spines = []
			
			if i == 0:
				internal_spines = [bottom_spine]
			elif i == self.num_rows-1:
				internal_spines = [top_spine]
			else:
				internal_spines = [top_spine, bottom_spine]

			for spine in internal_spines:
				spine.set_linewidth(internal_spine_width)
				spine.set_color('#b0a996')

	def clear_axes(self):
		"""Clear all axes in the plot"""
		# Clear chromatin axes
		for ax in self.chromatin_axes:
			plt.sca(ax)
			plt.cla()
		
		# Clear annotation axis
		plt.sca(self.annotation_axis)
		plt.cla()
		
		# Clear RNA pileup axis
		plt.sca(self.rna_pileup_axis)
		plt.cla()
			
		# Clear cell cycle axis
		plt.sca(self.cell_cycle_axis)
		plt.cla()

		self._initialize_axes()
		
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
		"""Plot image data on axis with highlighting support"""
		if vmax is None: 
			vmax = self.vmax
		ax.imshow(img, aspect='auto', cmap=cmap, origin='lower', interpolation='none',
			vmin=vmin, vmax=vmax, extent=[self.span[0], self.span[1], 0, 260])

		from src.plot_helpers import plot_rect2

		# Plot highlight bins if any
		for color, bin_tuple in self.highlight_bins:
			x_bp = bin_tuple[0], bin_tuple[2]
			y_bp = bin_tuple[1]+5, bin_tuple[3]-5 # Inset slightly for visuals

			plot_rect2(ax, x_bp[0], y_bp[0], 
					   x_bp[1], y_bp[1], alpha=0.8,
				lw=0.75, fill=False, edgecolor=color)

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

	def _plot_cell_cycle_annotations(self, branch_type):
		"""
		Plot cell cycle phase annotations with colored rectangles and text
		
		Parameters:
		-----------
		branch_type : str
			The branch type being plotted
		"""
		# Constants for positioning
		bar_width = 1.0
		bar_x = 0.0
		text_x = 0.5  # Centered
		
		ax = self.cell_cycle_axis
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

	def _plot_mean_mother_daughter_chromatin(self):
		"""Plot the average of mother and daughter chromatin data"""
		
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

			ax = self.chromatin_axes[row_idx]
			self.plot_im(ax, mean_img)

		self._plot_s_g2m_chromatin()

	def _plot_s_g2m_chromatin(self):
		"""Plot S and G2M phase chromatin data"""
		s_indices = get_sample_indices(self.config1, self.num_s_rows, 'S')
		g2m_indices = get_sample_indices(self.config1, self.num_g2m_rows, 'G2M')
		
		# Plot S phase chromatin
		for row_idx, idx in enumerate(s_indices):
			ax = self.chromatin_axes[row_idx + self.num_g1_rows]
			self.plot_im(ax, self.chromatin_F[idx])

		# Plot G2M phase chromatin
		for row_idx, idx in enumerate(g2m_indices):
			ax = self.chromatin_axes[row_idx + self.num_g1_rows + self.num_s_rows]
			self.plot_im(ax, self.chromatin_F[idx])

	def _plot_chromatin_branch(self, g1_phase):
		"""Plot chromatin data for a single branch"""
		
		# Get sampled indices for each phase
		g1_indices = get_sample_indices(self.config1, self.num_g1_rows, g1_phase)
	
		# Plot G1 phase chromatin
		for row_idx, idx in enumerate(g1_indices):
			ax = self.chromatin_axes[row_idx]
			self.plot_im(ax, self.chromatin_F[idx])

		self._plot_s_g2m_chromatin()

	def _plot_difference_mother_daughter_chromatin(self):
		"""Plot difference between daughter and mother chromatin data"""
		
		# Get sampled indices for each phase
		dg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'DG1')
		cg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'CG1')

		# Plot G1 phase chromatin difference
		vmax = 5
		eps = 1e-5

		for row_idx in range(len(dg1_indices)):
			ax = self.chromatin_axes[row_idx]

			dg1_dat = self.chromatin_F[dg1_indices[row_idx]]
			cg1_dat = self.chromatin_F[cg1_indices[row_idx]]
			diff = np.log2((dg1_dat+eps)/(cg1_dat+eps))
			self.plot_im(ax, diff, vmin=-vmax, vmax=vmax, cmap='RdBu_r')

		# Don't plot S/G2M for difference (only G1 phase makes sense for mother/daughter comparison)

	def _plot_chromatin_for_branch_type(self, branch_type):
		"""Plot chromatin data based on branch type"""
		if branch_type == "recovery":
			self._plot_chromatin_branch('RG1')
		elif branch_type == "mother":
			self._plot_chromatin_branch('CG1')
		elif branch_type == "daughter":
			self._plot_chromatin_branch('DG1')
		elif branch_type == "mean_mother_daughter":
			self._plot_mean_mother_daughter_chromatin()
		elif branch_type == "difference_mother_daughter":
			self._plot_difference_mother_daughter_chromatin()
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
		self.orf_plotter.set_span_chrom(span, chrom)
		self.rna_plotter.set_chrom_span(chrom, span, replicate='combined') 
		# todo only first replicate, plot the average of 1 and 2

	def _plot_annotations(self):
		"""Plot gene annotations on the annotation axis"""
		self.orf_plotter.plot_orf_annotations(self.annotation_axis)

	def _plot_rna_pileup(self):
		"""Plot RNA pileup data (placeholder for now)"""
		ax = self.rna_pileup_axis
		self.rna_plotter.plot_pileup(ax=ax, mode='minmax')
		
		ax.set_xticks([])
		ax.set_yticks([])

	def plot(self):
		"""Plot the chromatin data for the specified branch"""
		
		# Validate that required data is available
		if self.chromatin_F is None:
			raise ValueError("Chromatin data not set. Call set_chromatin_data first.")
		
		# Clear the axes for redraw
		self.clear_axes()

		# Plot gene annotations
		self._plot_annotations()

		# Plot RNA pileup (placeholder for now)
		self._plot_rna_pileup()

		# Plot cell cycle annotations
		self._plot_cell_cycle_annotations(self.branch_type)
		
		# Plot chromatin data
		self._plot_chromatin_for_branch_type(self.branch_type)

		# Set title
		title = self.title
		if title is None:
			title = f"chr{self.chrom}, {self.span[0]}-{self.span[1]} ({self.branch_type})"
		plt.suptitle(title, fontsize=24, y=1, fontweight='demi')

		# For the last row of the chromatin axes, show ticks for positions
		minor_xticks = np.arange(self.span[0], self.span[1]+200, 200)
		major_xticks = np.arange(self.span[0], self.span[1]+1000, 1000)

		ax = self.chromatin_axes[-1]

		# Set the tick mark locations
		ax.set_xticks(minor_xticks, minor=True)
		ax.set_xticks(major_xticks, minor=False)

		# Formatting
		ax.tick_params(axis='x', which='major', length=5, width=1.25, labelbottom=False)
		ax.tick_params(axis='x', which='minor', length=2, width=1, labelbottom=False)
		ax.set_xlim(*self.span)

		return self.fig


def create_single_branch_chromatin_layout(
	n_rows=7,  # Number of rows in chromatin data (excluding annotation)
	figsize=(6, 5),  # Figure size
	cell_cycle_annotations_width=0.07,  # Width of cell cycle annotations plot relative to chromatin
	top_margin=0.95,  # Top margin for titles
	bottom_margin=0.1,  # Bottom margin
	height_ratios=None,  # Optional custom height ratios for rows
	annotation_height=1.2,  # Height of annotation row relative to data rows
	annotation_spacing=0.2,  # Height of spacing between annotation and rna_pileup rows
	rna_pileup_height=1.0,  # Height of rna_pileup row relative to data rows
	rna_pileup_spacing=0.2,  # Height of spacing between rna_pileup and chromatin data rows
):
	"""
	Creates a layout for single-branch chromatin data visualization with gene annotations,
	RNA pileup, and cell cycle indicators.
	
	Returns:
		fig: matplotlib figure
		chromatin_axes: list of axes for chromatin data rows
		annotation_axis: single axis for gene annotations
		rna_pileup_axis: single axis for RNA pileup data
		cell_cycle_axis: single axis for cell cycle annotations
	"""
	# Create figure
	fig = plt.figure(figsize=figsize)
	
	# Column structure: cell cycle + chromatin
	width_ratios = [cell_cycle_annotations_width, 1]
	
	# Row structure: annotation + spacing + rna_pileup + spacing + data rows
	if height_ratios is None:
		data_height_ratios = [1] * n_rows
	else:
		data_height_ratios = height_ratios
	
	total_height_ratios = [
		annotation_height, 
		annotation_spacing, 
		rna_pileup_height, 
		rna_pileup_spacing
	] + data_height_ratios
	
	# Create GridSpec
	gs = gridspec.GridSpec(
		n_rows + 4,  # Add 4 for annotation, annotation_spacing, rna_pileup, rna_pileup_spacing
		2,  # 2 columns: cell cycle + chromatin
		width_ratios=width_ratios,
		height_ratios=total_height_ratios,
		hspace=0,
		wspace=0
	)
	
	# Create annotation axis (spans only chromatin column)
	annotation_axis = fig.add_subplot(gs[0, 1])
	
	# Create RNA pileup axis (spans only chromatin column) 
	rna_pileup_axis = fig.add_subplot(gs[2, 1])
	
	# Create cell cycle axis (spans all data rows in cell cycle column)
	cell_cycle_axis = fig.add_subplot(gs[4:, 0])
	
	# Create chromatin axes (one for each data row in chromatin column)
	chromatin_axes = []
	for row in range(n_rows):
		ax = fig.add_subplot(gs[row + 4, 1])
		chromatin_axes.append(ax)
	
	# Adjust layout
	plt.subplots_adjust(
		top=top_margin,
		bottom=bottom_margin
	)
	
	return fig, chromatin_axes, annotation_axis, rna_pileup_axis, cell_cycle_axis
