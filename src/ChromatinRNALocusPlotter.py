
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.plot_helpers import hide_spines
from src.tf_sites import TFBindingSites
from src.plot_helpers import color_for_key
from src.config_utils import get_sample_indices
from src.orf_plotter import load_default_orf_plotter
from src.rna_pileup_plotter import RNASeqPileupPlotter
from src.figure_configs import tf_colors

internal_spine_color = '#b0a996'

class SingleBranchChromatinPlotter:
	"""
	Class to plot the chromatin context for a single branch.
	
	This class focuses on plotting chromatin data for one specific branch type
	(recovery, mother, daughter, mean, or difference) with gene annotations,
	RNA pileup, and cell cycle annotations.
	"""

	def __init__(self, outdir, config1, branch_type="mother", figsize=(6, 5), title=None,
		rna_plotter=None, deconvolved_tpm_plotter=None, plot_index_labels=True,
		tfs=[]):
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
		self.add_genomic_scale = False
		self.add_xticks = True
		self.tfs = tfs

		# Initialize legend storage
		self.tf_legend_items = []
		self.tf_legend_labels = []

		# Right side index labels
		self.plot_index_labels = plot_index_labels

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
		self.num_rows = 8
		self.vmax = 40
		self.num_g1_rows = 4
		self.num_s_rows = 2
		self.num_g2m_rows = self.num_rows - self.num_g1_rows - self.num_s_rows

		# Create the layout using our new single-branch layout function
		self.fig, self.chromatin_axes, self.annotation_axis, self.rna_pileup_axis, \
		self.cell_cycle_axis = \
			create_single_branch_chromatin_layout(
				figsize=figsize, 
				n_rows=self.num_rows
			)
			
		self.orf_plotter = load_default_orf_plotter(outdir)
		self.deconvolved_tpm_plotter = deconvolved_tpm_plotter

		if rna_plotter is None:
			self.rna_plotter = RNASeqPileupPlotter(outdir)
		else:
			self.rna_plotter = rna_plotter

		self.rna_plotter.ylim = 10

		# Initialize the axes
		self._initialize_axes()

		self.binding_sites = TFBindingSites()

	def all_data_axes(self):
		all_data_axes = [self.annotation_axis, self.rna_pileup_axis] + self.chromatin_axes
		return all_data_axes

		
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
				spine.set_color(internal_spine_color)

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
		ax.set_xlim(self.span[0], self.span[1])
		ax.set_ylim(0, 260)

		from src.plot_helpers import plot_rect2

		# Plot highlight bins if any
		for color, bin_tuple in self.highlight_bins:
			x_bp = bin_tuple[0], bin_tuple[2]
			y_bp = bin_tuple[1]+5, bin_tuple[3]-5 # Inset slightly for visuals

			plot_rect2(ax, x_bp[0], y_bp[0], 
					   x_bp[1], y_bp[1], alpha=0.8,
				lw=0.75, fill=False, edgecolor=color)

		# Plot transcription factor binding sites
		for tf in self.current_binding_sites.tf.values:

			# Plot selected tfs
			if self.tfs == 'all' or tf in self.tfs:

				sites_to_plot = self.current_binding_sites[self.current_binding_sites.tf == tf]

				color = tf_colors[tf] if tf in tf_colors else tf_colors['other']

				scatter = ax.scatter(sites_to_plot.start, [30]*len(sites_to_plot), 
					color=color, marker='^', s=20, zorder=100)

				if not tf in self.tf_legend_labels:
					self.tf_legend_items.append(scatter)
					self.tf_legend_labels.append(tf)

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
		text_x = 0.6 # Centered, shifted slightly
		
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


		config = self.config1

		t_indices = config.get_Hpositions_for_branch('t')
		cg1_indices = config.get_Hpositions_for_phase('CG1')
		s_indices = config.get_Hpositions_for_phase('S')
		g2m_indices = config.get_Hpositions_for_phase('G2M')

		g1_num_label = f"n={len(cg1_indices)}"
		s_num_label = f"n={len(s_indices)}"
		g2m_num_label = f"n={len(g2m_indices)}"

		# Add vertical text labels
		# G1 phase text
		annotation_fontsize = 12
		num_fontsize = 10

		def _plot_annotation_text(x, y, label, fontsize, ha='center'):
			ax.text(x, y, label, fontsize=fontsize,
					rotation=90, ha=ha, va='center', color='white',
					fontdict={'fontname': 'Open Sans'})

		g1_center = self.num_g1_rows / 2
		s_center = self.num_g1_rows + (self.num_s_rows / 2)
		g2m_center = self.num_g1_rows + self.num_s_rows + (self.num_g2m_rows / 2)
		label_between_padding = 0.05
		pad_2 = label_between_padding/2

		# Phase label names
		if self.plot_index_labels:
			phase_label_x_position = text_x-pad_2
			number_label_x_position = text_x+pad_2
			phase_ha = 'right'
		else:
			phase_label_x_position = text_x
			phase_ha = 'center'

		_plot_annotation_text(phase_label_x_position, g1_center, self.map_phase_name[g1_phase], annotation_fontsize, 
			ha=phase_ha)
		_plot_annotation_text(phase_label_x_position, s_center, self.map_phase_name['S'], annotation_fontsize, 
			ha=phase_ha)
		_plot_annotation_text(phase_label_x_position, g2m_center, self.map_phase_name['G2M'], annotation_fontsize, 
			ha=phase_ha)

		# Number of indices per phase
		if self.plot_index_labels:
			_plot_annotation_text(number_label_x_position, g1_center, g1_num_label, num_fontsize,
				ha='left')
			_plot_annotation_text(number_label_x_position, s_center, s_num_label, num_fontsize,
				ha='left')
			_plot_annotation_text(number_label_x_position, g2m_center, g2m_num_label, num_fontsize,
				ha='left')
		
		# Set axis limits
		ax.set_xlim(0, 1)
		ax.set_ylim(self.num_rows, 0)

	def plot_deconvolved_tpm_if_needed(self, ax, indices):
		if self.deconvolved_tpm_plotter is not None:
			self.deconvolved_tpm_plotter.plot_mean_transcripts_for_time_indices(ax, indices)

	def _plot_mean_mother_daughter_chromatin(self):
		"""Plot the average of mother and daughter chromatin data"""
		
		dg1_indices, dg1_abs = get_sample_indices(self.config1, self.num_g1_rows, 
			'DG1', return_absolute=True)
		mg1_indices, mg1_abs = get_sample_indices(self.config1, self.num_g1_rows, 
			'CG1', return_absolute=True)

		total_branch_indices = len(self.config1.get_Hpositions_for_branch('t'))

		from src.plot_helpers import _plot_index_label
		
		# Plot G1 phase chromatin
		for row_idx in range(len(mg1_indices)):
			# Take the average of the MG1 and DG1 image
			dg1_idx = dg1_indices[row_idx]
			mg1_idx = mg1_indices[row_idx]
			dg1_img = self.chromatin_F[dg1_idx]
			mg1_img = self.chromatin_F[mg1_idx]
			mean_img = (dg1_img+mg1_img)/2.

			# Plot the chromatin image for the index
			ax = self.chromatin_axes[row_idx]
			self.plot_im(ax, mean_img)

			# Plot the deconvolved TPM
			self.plot_deconvolved_tpm_if_needed(ax, [dg1_idx, mg1_idx])

			# Add index label
			if self.plot_index_labels:
				_plot_index_label(ax, self.span[1]+50, 130, dg1_abs[row_idx]+1, total_branch_indices)

			ax.set_ylim(0, 260)

		self._plot_s_g2m_chromatin()

	def _plot_s_g2m_chromatin(self):
		"""Plot S and G2M phase chromatin data"""
		s_indices, s_abs = get_sample_indices(self.config1, self.num_s_rows, 'S', True)
		g2m_indices, g2m_abs = get_sample_indices(self.config1, self.num_g2m_rows, 'G2M', True)

		total_g1 = len(self.config1.get_Hpositions_for_phase('CG1'))
		total_s = len(self.config1.get_Hpositions_for_phase('S'))
		total_branch_indices = len(self.config1.get_Hpositions_for_branch('t'))
		
		from src.plot_helpers import _plot_index_label

		# Plot S phase chromatin
		for row_idx, idx in enumerate(s_indices):
			ax = self.chromatin_axes[row_idx + self.num_g1_rows]
			self.plot_im(ax, self.chromatin_F[idx])

			if self.plot_index_labels:
				_plot_index_label(ax, self.span[1]+50, 130, s_abs[row_idx]+1+total_g1, 
					total_branch_indices)

			self.plot_deconvolved_tpm_if_needed(ax, [idx])
			ax.set_ylim(0, 260)

		# Plot G2M phase chromatin
		for row_idx, idx in enumerate(g2m_indices):
			ax = self.chromatin_axes[row_idx + self.num_g1_rows + self.num_s_rows]
			self.plot_im(ax, self.chromatin_F[idx])

			if self.plot_index_labels:
				_plot_index_label(ax, self.span[1]+50, 130, g2m_abs[row_idx]+1+total_g1+total_s, 
					total_branch_indices)

			self.plot_deconvolved_tpm_if_needed(ax, [idx])
			ax.set_ylim(0, 260)

	def _plot_chromatin_branch(self, g1_phase):
		"""Plot chromatin data for a single branch"""

		from src.plot_helpers import _plot_index_label
		
		# Get sampled indices for each phase
		g1_indices = get_sample_indices(self.config1, self.num_g1_rows, g1_phase)
		absolute_indices = np.arange(len(g1_indices)) # Indices for plotting labels
		total_branch_indices = len(self.config1.get_Hpositions_for_branch('t'))
	
		# Plot G1 phase chromatin
		for row_idx, idx in enumerate(g1_indices):
			ax = self.chromatin_axes[row_idx]
			self.plot_im(ax, self.chromatin_F[idx])

			# Plot the deconvolved TPM
			self.plot_deconvolved_tpm_if_needed(ax, [idx])

			# Add index label
			if self.plot_index_labels:
				_plot_index_label(ax, self.span[1]+50, 130, absolute_indices[row_idx]+1, 
					total_branch_indices)

		self._plot_s_g2m_chromatin()

	def _plot_difference_mother_daughter_chromatin(self):
		"""Plot difference between daughter and mother chromatin data"""
		
		# Get sampled indices for each phase
		dg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'DG1')
		cg1_indices = get_sample_indices(self.config1, self.num_g1_rows, 'CG1')
		s_indices = get_sample_indices(self.config1, self.num_s_rows, 'S')
		g2m_indices = get_sample_indices(self.config1, self.num_g2m_rows, 'G2M')

		# Plot G1 phase chromatin difference
		vmax = 5
		eps = 1e-5

		for row_idx in range(len(dg1_indices)):
			ax = self.chromatin_axes[row_idx]

			dg1_dat = self.chromatin_F[dg1_indices[row_idx]]
			cg1_dat = self.chromatin_F[cg1_indices[row_idx]]
			diff = np.log2((dg1_dat+eps)/(cg1_dat+eps))
			self.plot_im(ax, diff, vmin=-vmax, vmax=vmax, cmap='RdBu_r')

		# Plot zeros for the remaining indices

		# Plot S phase chromatin
		zeros = np.zeros_like(diff)
		for row_idx, idx in enumerate(s_indices):
			ax = self.chromatin_axes[row_idx + self.num_g1_rows]
			self.plot_im(ax, zeros, vmin=-vmax, vmax=vmax, cmap='RdBu_r')

		# Plot G2M phase chromatin
		for row_idx, idx in enumerate(g2m_indices):
			ax = self.chromatin_axes[row_idx + self.num_g1_rows + self.num_s_rows]
			self.plot_im(ax, zeros, vmin=-vmax, vmax=vmax, cmap='RdBu_r')


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

		if self.add_genomic_scale:
			from src.plot_helpers import add_im_genomic_scale_legend

			ax = self.chromatin_axes[-1]
			add_im_genomic_scale_legend(ax, self.span[0], 1000, legend_y=-80)


		# Add legend if necessary
		if len(self.tf_legend_items) > 0:
			ax = self.chromatin_axes[-1]
			ax.legend(self.tf_legend_items, self.tf_legend_labels,
				bbox_to_anchor=(-0.1, -0.2),
				loc='upper left',
				ncol=10,
				fontsize=13,
				handletextpad=0.,
				frameon=False)

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
		self.rna_plotter.set_chrom_span(chrom, span, replicate='combined') 

		rossi_sites = self.binding_sites.all_rossi_tf_dfs
		self.current_binding_sites = rossi_sites[(rossi_sites.chr == chrom) &
			   (rossi_sites.start > span[0]) & 
			   (rossi_sites.start < span[1])]

		if self.deconvolved_tpm_plotter is not None:
			self.deconvolved_tpm_plotter.set_chrom_span(chrom, span)


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
			from src.read_bam import _toRoman
			title = f"chr{_toRoman(self.chrom)} {self.span[0]}...{self.span[1]}"

		self.annotation_axis.set_title(title, fontsize=24, fontweight='demi', pad=13)

		self.rna_pileup_axis.set_ylabel("Experimental\nRNA pileup", fontsize=12,
			fontweight='demi', labelpad=5)

		# Labe the deconvolution section
		self.chromatin_axes[0].set_title("Deconvolved chromatin and transcription",
			fontsize=18, fontweight='demi', pad=10)

		# For the last row of the chromatin axes, show ticks for positions
		minor_xticks = np.arange(self.span[0], self.span[1]+200, 200)
		major_xticks = np.arange(self.span[0], self.span[1]+1000, 1000)

		ax = self.chromatin_axes[-1]

		# Set the tick mark locations
		ax.set_xticks(minor_xticks, minor=True)
		ax.set_xticks(major_xticks, minor=False)

		# Formatting
		ax.tick_params(axis='x', which='major', length=5, width=1.25, labelbottom=self.add_xticks)
		ax.tick_params(axis='x', which='minor', length=2, width=1, labelbottom=False)
		ax.set_xlim(*self.span)

		return self.fig


def create_single_branch_chromatin_layout(
	n_rows=7,  # Number of rows in chromatin data (excluding annotation+rna)
	figsize=(6, 5),  # Figure size
	cell_cycle_annotations_width=0.07,  # Width of cell cycle annotations plot relative to chromatin
	top_margin=0.95,  # Top margin for titles
	bottom_margin=0.1,  # Bottom margin
	height_ratios=None,  # Optional custom height ratios for rows
	annotation_height=1.0,  # Height of annotation row relative to data rows
	annotation_spacing=0.2,  # Height of spacing between annotation and rna_pileup rows
	rna_pileup_height=1.6,  # Height of rna_pileup row relative to data rows
	rna_pileup_spacing=0.6,  # Height of spacing between rna_pileup and chromatin data rows
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
