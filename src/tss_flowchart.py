import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class GeneStats:
	"""Data class to hold gene analysis statistics"""
	total_genes: int
	with_park_tss: int
	without_park_tss: int
	with_park_rna_yes: int
	with_park_rna_no: int
	without_park_rna_yes: int
	without_park_rna_no: int
	
	def __post_init__(self):
		pass

@dataclass 
class StyleConfig:
	"""Configuration for visual styling"""
	colors: Dict[str, str] = None
	box_style: str = "round,pad=0.1"
	arrow_width: float = 2.0
	highlight_arrow_width: float = 3.0
	font_sizes: Dict[str, int] = None
	
	def __post_init__(self):
		if self.colors is None:
			self.colors = {
				'total': '#ddd',
				'park_yes': '#ddd', 
				'park_no': '#ddd',
				'rna_yes': '#ddd',
				'rna_no': '#ddd',
				'park_yes_rna_yes': plt.cm.Oranges(0.25),
				'park_no_rna_yes': plt.cm.Blues(0.25)
			}
		if self.font_sizes is None:
			self.font_sizes = {
				'title': 24,
				'level1': 12,
				'level2': 12, 
				'level3': 11,
				'hist_title': 14,
				'hist_label': 12
			}

@dataclass
class LayoutConfig:
	"""Configuration for layout positioning"""
	flowchart_bounds: Tuple[float, float, float, float] = (0, 16, 0, 10)  # xmin, xmax, ymin, ymax (wider)
	level_y_positions: List[float] = None
	box_dimensions: Dict[str, Tuple[float, float]] = None
	grid_ratios: Dict[str, List[float]] = None
	level3_spacing: float = 0.1  # spacing between level 3 boxes
	
	def __post_init__(self):
		if self.level_y_positions is None:
			self.level_y_positions = [8.5, 6.0, 3.5]  # y positions for levels 1, 2, 3
		if self.box_dimensions is None:
			self.box_dimensions = {
				'level1': (3.0, 1.2),
				'level2': (3.0, 1.0), 
				'level3': (2., 1.0)  # slightly smaller level 3 boxes
			}
		if self.grid_ratios is None:
			self.grid_ratios = {
				'height': [1, 1, 0.3, 1.5],
				'width': [1, 1, 1, 1]
			}

class FlowchartBox:
	"""Helper class for creating flowchart boxes"""
	
	def __init__(self, x: float, y: float, width: float, height: float, 
				 text: str, style_config: StyleConfig, color_key: str, 
				 is_highlighted: bool = False):
		self.x = x
		self.y = y 
		self.width = width
		self.height = height
		self.text = text
		self.style_config = style_config
		self.color_key = color_key
		self.is_highlighted = is_highlighted
		
	def add_to_axes(self, ax):
		"""Add this box to the given axes"""
		linewidth = 2 if self.is_highlighted else 1
		box = patches.FancyBboxPatch(
			(self.x, self.y), self.width, self.height,
			boxstyle=self.style_config.box_style,
			facecolor=self.style_config.colors[self.color_key],
			edgecolor='black', 
			linewidth=linewidth,
			zorder=3,
		)
		ax.add_patch(box)
		
		# Determine text color and weight
		text_color = 'black'
		font_weight = 'demi'
		
		ax.text(self.x + self.width/2, self.y + self.height/2, self.text,
				ha='center', va='center', 
				fontsize=self._get_font_size(),
				fontweight=font_weight, 
				color=text_color,
				zorder=4)
	
	def _get_font_size(self):
		"""Determine font size based on box level"""
		if 'Total' in self.text:
			return self.style_config.font_sizes['level1']
		elif any(x in self.text for x in ['Park TSS', 'Without Park']):
			return self.style_config.font_sizes['level2']
		else:
			return self.style_config.font_sizes['level3']
	
	def get_center(self):
		"""Get the center coordinates of the box"""
		return (self.x + self.width/2, self.y + self.height/2)
	
	def get_bottom_center(self):
		"""Get the bottom center coordinates of the box"""
		return (self.x + self.width/2, self.y)
	
	def get_top_center(self):
		"""Get the top center coordinates of the box"""
		return (self.x + self.width/2, self.y + self.height)
	
	def get_left_center(self):
		"""Get the left center coordinates of the box"""
		return (self.x, self.y + self.height/2)
	
	def get_right_center(self):
		"""Get the right center coordinates of the box"""
		return (self.x + self.width, self.y + self.height/2)

class FlowchartArrow:
	"""Helper class for creating arrows"""
	
	def __init__(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float],
				 style_config: StyleConfig, color: str = 'black', is_highlighted: bool = False):
		self.start_pos = start_pos
		self.end_pos = end_pos
		self.style_config = style_config
		self.color = color
		self.is_highlighted = is_highlighted
	
	def add_to_axes(self, ax):
		"""Add this arrow to the given axes"""
		linewidth = self.style_config.highlight_arrow_width if self.is_highlighted else self.style_config.arrow_width
		ax.annotate('', xy=self.end_pos, xytext=self.start_pos,
				   arrowprops=dict(arrowstyle='->', lw=linewidth, color='black',
				   	zorder=1))

class GeneAnalysisFlowchart:
	"""Main class for creating gene analysis flowcharts"""
	
	def __init__(self, stats: GeneStats, style_config: StyleConfig = None, 
				 layout_config: LayoutConfig = None):
		self.stats = stats
		self.style_config = style_config or StyleConfig()
		self.layout_config = layout_config or LayoutConfig()
		self.boxes = []
		self.arrows = []
	
	def _create_boxes(self):
		"""Create all flowchart boxes based on the statistics"""
		xmin, xmax, ymin, ymax = self.layout_config.flowchart_bounds
		center_x = (xmin + xmax) / 2
		
		# Level 1: Total genes
		box_w, box_h = self.layout_config.box_dimensions['level1']
		level1_y = self.layout_config.level_y_positions[0]
		
		total_box = FlowchartBox(
			center_x - box_w/2, level1_y, box_w, box_h,
			f'Total Genes\n(non-dubious)\n{self.stats["total_genes"]}',
			self.style_config, 'total', is_highlighted=True
		)
		self.boxes.append(total_box)
		
		# Level 2: Park TSS split
		level2_y = self.layout_config.level_y_positions[1]
		left_x = center_x - box_w - 1
		right_x = center_x + 1
		
		park_yes_box = FlowchartBox(
			left_x, level2_y, box_w, box_h,
			f'With Park TSS\n{self.stats["genes_with_park"]:,}',
			self.style_config, 'park_yes', is_highlighted=True
		)
		self.boxes.append(park_yes_box)
		
		park_no_box = FlowchartBox(
			right_x, level2_y, box_w, box_h,
			f'Without Park TSS\n{self.stats["genes_without_park"]:,}',
			self.style_config, 'park_no', is_highlighted=True
		)
		self.boxes.append(park_no_box)
		
		# Level 3: RNA splits  
		level3_y = self.layout_config.level_y_positions[2]
		box_w3, box_h3 = self.layout_config.box_dimensions['level3']
		spacing = self.layout_config.level3_spacing
		
		# Calculate positions systematically to avoid overlap
		# Left side (under "With Park TSS")
		left_rna_yes_x = left_x - box_w3/2
		left_rna_no_x = left_x + box_w/2 + spacing
		
		# Right side (under "Without Park TSS") 
		right_rna_yes_x = right_x - box_w3/2
		right_rna_no_x = right_x + box_w/2 + spacing
		
		rna_boxes_data = [
			(left_rna_yes_x, f'With RNA calls\n{self.stats["genes_with_park_and_rna"]:,}', 'park_yes_rna_yes', True),
			(left_rna_no_x, f'Without RNA calls\n{self.stats["genes_with_park_no_rna"]:,}', 'rna_no', False),
			(right_rna_yes_x, f'With RNA calls\n{self.stats["genes_without_park_with_rna"]:,}', 'park_no_rna_yes', True),
			(right_rna_no_x, f'Without RNA calls\n{self.stats["genes_without_park_no_rna"]:,}', 'rna_no', False)
		]
		
		for x_pos, text, color_key, highlighted in rna_boxes_data:
			box = FlowchartBox(x_pos, level3_y, box_w3, box_h3, text, 
							 self.style_config, color_key, highlighted)
			self.boxes.append(box)
	
	def _create_arrows(self):
		"""Create arrows connecting the boxes with proper edge positioning"""
		# Level 1 to Level 2 arrows
		total_box = self.boxes[0]
		park_yes_box = self.boxes[1] 
		park_no_box = self.boxes[2]
		
		# Arrows from bottom of total box to top of park boxes
		total_bottom = total_box.get_bottom_center()
		park_yes_top = park_yes_box.get_top_center()
		park_no_top = park_no_box.get_top_center()
		
		self.arrows.extend([
			FlowchartArrow(total_bottom, park_yes_top, self.style_config),
			FlowchartArrow(total_bottom, park_no_top, self.style_config)
		])
		
		# Level 2 to Level 3 arrows
		for i in range(3, 7):  # RNA boxes indices
			parent_idx = 1 if i < 5 else 2  # boxes 3,4 connect to box 1; boxes 5,6 to box 2
			parent_box = self.boxes[parent_idx]
			child_box = self.boxes[i]
			
			# Arrow from bottom of parent to top of child
			parent_bottom = parent_box.get_bottom_center()
			child_top = child_box.get_top_center()
			
			self.arrows.append(FlowchartArrow(parent_bottom, child_top, self.style_config))
		
		# Special arrows to histograms (only from RNA yes boxes)
		rna_yes_boxes = [self.boxes[3], self.boxes[5]]  # boxes with RNA calls
		histogram_y = 1.5
		
		for box in rna_yes_boxes:
			# Arrow from bottom of RNA yes box to histogram area
			start_pos = box.get_bottom_center()
			end_pos = (start_pos[0], histogram_y)
			arrow = FlowchartArrow(start_pos, end_pos, self.style_config, 
								 color=self.style_config.colors['rna_yes'])
			self.arrows.append(arrow)
	
	def create_figure(self, data_with_park_tss, data_without_park_tss, 
					 figsize=(14, 10), title="Gene TSS updates"):
		"""Create the complete figure with flowchart and histograms"""
		
		# Create boxes and arrows
		self._create_boxes()
		self._create_arrows()
		
		# Set up figure and grid
		fig = plt.figure(figsize=figsize)
		gs = GridSpec(4, 4, figure=fig,
					  height_ratios=self.layout_config.grid_ratios['height'],
					  width_ratios=self.layout_config.grid_ratios['width'],
					  hspace=0.0, wspace=0.3)
		
		# Create axes
		ax_flowchart = fig.add_subplot(gs[:3, :])
		ax_hist_with = fig.add_subplot(gs[3, :2])
		ax_hist_without = fig.add_subplot(gs[3, 2:])
		
		# Set up flowchart
		xmin, xmax, ymin, ymax = self.layout_config.flowchart_bounds
		ax_flowchart.set_xlim(xmin, xmax)
		ax_flowchart.set_ylim(ymin, ymax)
		ax_flowchart.axis('off')
		
		for arrow in self.arrows:
			arrow.add_to_axes(ax_flowchart)

		# Add all boxes and arrows to flowchart
		for box in self.boxes:
			box.add_to_axes(ax_flowchart)
		
		# Create histograms
		self._create_histograms(ax_hist_with, ax_hist_without, 
							   data_with_park_tss, data_without_park_tss)
		
		# Add main title
		fig.suptitle(title, fontsize=self.style_config.font_sizes['title'], 
					fontweight='demi', y=0.95)
		
		axes = {
			'flowchart': ax_flowchart,
			'hist_with_park': ax_hist_with,
			'hist_without_park': ax_hist_without
		}
		
		return fig, axes
	
	def _create_histograms(self, ax_with, ax_without, data_with, data_without):
		"""Create the histogram plots"""
		hist_color = self.style_config.colors['rna_yes']
		
		# Left histogram (with Park TSS)
		bins = np.arange(-2000, 2000, 100)
		ax_with.hist(data_with, bins=bins, color=plt.cm.Oranges(0.5), alpha=0.8, 
					edgecolor='black', linewidth=0.5)
		ax_with.set_title(f'Genes with Park TSS,\nwith RNA calls, (n={len(data_with):,})',
						 fontsize=self.style_config.font_sizes['hist_title'], 
						 fontweight='demi', color='black')
		ax_with.set_xlabel('Update difference', fontsize=self.style_config.font_sizes['hist_label'])
		ax_with.set_ylabel('Frequency', fontsize=self.style_config.font_sizes['hist_label'])
		ax_with.set_xlim(-2000, 2000)
		ax_with.set_ylim(0, 2200)

		update_cutoff = 200
		ax_with.axvline(-update_cutoff, c='red')
		ax_with.axvline(update_cutoff, c='red')

		num_updated_park = len(data_with[(data_with > -update_cutoff) & (data_with < update_cutoff)])
		y = 2000
		y_cutoffs = 1000
		ax_with.text(0, y, f"{num_updated_park}", ha='center', 
			fontsize=14, color='red')
		ax_with.text(update_cutoff+20, y_cutoffs, f"+{update_cutoff}", 
			fontsize=14, color='red', ha='left')
		ax_with.text(-update_cutoff-20, y_cutoffs, f"-{update_cutoff}",
			fontsize=14, color='red', ha='right')
		
		# Right histogram (without Park TSS)
		ax_without.hist(data_without, bins=bins, color=plt.cm.Blues(0.5), alpha=0.8,
					   edgecolor='black', linewidth=0.5)
		ax_without.set_title(f'Genes without Park TSS,\nwith RNA calls, (n={len(data_without)})',
							fontsize=self.style_config.font_sizes['hist_title'],
							fontweight='demi', color='black')
		ax_without.set_xlabel('Update difference', fontsize=self.style_config.font_sizes['hist_label'])
		ax_without.set_ylabel('Frequency', fontsize=self.style_config.font_sizes['hist_label'])
		ax_without.set_xlim(-2000, 2000)
		ax_without.set_ylim(0, 80)
		ax_without.axvline(-update_cutoff, c='red')
		ax_without.axvline(update_cutoff, c='red')

		num_updated_sgd = len(data_without[(data_without > -update_cutoff) & (data_without < update_cutoff)])

		y = 70
		y_cutoffs = 20
		ax_without.text(0, y, f"{num_updated_sgd}", ha='center', 
			fontsize=14, color='red')
		ax_without.text(update_cutoff+20, y_cutoffs, f"+{update_cutoff}", 
			fontsize=14, color='red', ha='left')
		ax_without.text(-update_cutoff-20, y_cutoffs, f"-{update_cutoff}", 
			fontsize=14, color='red', ha='right')


# Convenience function to maintain backwards compatibility
def create_gene_analysis_figure(data_with_park_tss, data_without_park_tss,
							   stats: GeneStats = None, figsize=(14, 10), 
							   title="Gene TSS updates",
							   style_config: StyleConfig = None,
							   layout_config: LayoutConfig = None):
	"""
	Create a gene analysis flowchart with histograms.
	
	Parameters:
	-----------
	data_with_park_tss : array-like
		Data for genes with Park TSS and RNA calls
	data_without_park_tss : array-like
		Data for genes without Park TSS and RNA calls  
	stats : GeneStats, optional
		Gene statistics. If None, uses default values from your example.
	figsize : tuple
		Figure size
	title : str
		Main title
	style_config : StyleConfig, optional
		Custom styling configuration
	layout_config : LayoutConfig, optional  
		Custom layout configuration
	"""
	
	# Use default stats if none provided
	if stats is None:
		stats = GeneStats(
			total_genes=5774,
			with_park_tss=5206,
			without_park_tss=568,
			with_park_rna_yes=4001,
			with_park_rna_no=1205,
			without_park_rna_yes=260, 
			without_park_rna_no=308
		)
	
	flowchart = GeneAnalysisFlowchart(stats, style_config, layout_config)
	return flowchart.create_figure(data_with_park_tss, data_without_park_tss, 
								  figsize, title)
