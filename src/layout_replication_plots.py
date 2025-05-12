import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

class ReplicationSubplotLayout:
	def __init__(self, figsize=(10, 8), 
				 left_width=0.1,           # Width of left side plots relative to figure width
				 top_height=0.175,           # Height of top plot relative to figure height
				 bottom_height=0.15,        # Height of bottom plot relative to figure height
				 main_height_ratio=0.5,     # Ratio of top main plot to bottom main plot (0.5 means equal)
				 hspace_main=0.0,           # Vertical spacing between main heatmaps
				 hgap_left_main=0.05,       # Horizontal gap between left plots and main plots
				 vgap_top_main=0.05,        # Vertical gap between top plot and main plots
				 vgap_bottom_main=0.05):    # Vertical gap between bottom plot and main plots
		
		self.fig = plt.figure(figsize=figsize)
		
		# Calculate heights with gaps
		total_height = 1.0
		total_width = 1.0
		
		# Vertical proportions including gaps
		self.vgap_top_main = vgap_top_main
		self.vgap_bottom_main = vgap_bottom_main
		self.hgap_left_main = hgap_left_main
		
		# Calculate effective heights and widths accounting for gaps
		effective_height = total_height - vgap_top_main - vgap_bottom_main
		effective_width = total_width - hgap_left_main
		
		# Calculate proportional heights and widths
		self.top_prop = top_height / effective_height
		self.bottom_prop = bottom_height / effective_height
		self.main_prop = (effective_height - top_height - bottom_height) / effective_height
		
		self.left_prop = left_width / effective_width
		self.main_width_prop = (effective_width - left_width) / effective_width
		
		# Create the figure with a grid for overall layout
		self.fig = plt.figure(figsize=figsize)
		
		# Adjust figure margins to accommodate our custom layout
		self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
		
		# Create a 4x2 grid with appropriate height and width ratios
		main_area_height = 1.0 - self.top_prop - self.bottom_prop - self.vgap_top_main - self.vgap_bottom_main
		
		# Create height ratios: top, gap1, main1, main2, gap2, bottom
		height_ratios = [
			self.top_prop,             # Top plot
			self.vgap_top_main,        # Gap between top and main
			main_area_height * main_height_ratio,  # Top main plot
			main_area_height * (1-main_height_ratio),  # Bottom main plot
			self.vgap_bottom_main,     # Gap between main and bottom
			self.bottom_prop           # Bottom plot
		]
		
		# Create width ratios: left, gap, main
		width_ratios = [
			self.left_prop,            # Left plots
			self.hgap_left_main,       # Gap between left and main
			self.main_width_prop       # Main plots
		]
		
		# Create the overall grid
		self.gs = gridspec.GridSpec(6, 3, height_ratios=height_ratios, width_ratios=width_ratios)
		
		# Create subplots with explicit positioning
		self.ax_top = self.fig.add_subplot(self.gs[0, 2])
		self.ax_left_top = self.fig.add_subplot(self.gs[2, 0])
		self.ax_left_bottom = self.fig.add_subplot(self.gs[3, 0])
		self.ax_main_top = self.fig.add_subplot(self.gs[2, 2])
		self.ax_main_bottom = self.fig.add_subplot(self.gs[3, 2])
		self.ax_bottom = self.fig.add_subplot(self.gs[5, 2])

		for ax in self.get_axes().values():
			ax.set_xticks([])
			ax.set_yticks([])

		plt.subplots_adjust(hspace=0, wspace=0)
		
	def get_axes(self):
		"""Return all axes as a dictionary for easy access"""
		return {
			'top': self.ax_top,
			'left_top': self.ax_left_top,
			'left_bottom': self.ax_left_bottom,
			'main_top': self.ax_main_top,
			'main_bottom': self.ax_main_bottom,
			'bottom': self.ax_bottom
		}
		
	def show(self):
		"""Display the figure"""
		plt.show()
		
	def save(self, filename, dpi=300, **kwargs):
		"""Save the figure to a file"""
		self.fig.savefig(filename, dpi=dpi, **kwargs)


import matplotlib.pyplot as plt
import numpy as np

class FlexibleRowLayout:
	def __init__(self, 
				 n_plots=5,                     # Number of plots
				 widths=None,                   # Relative widths of each plot
				 heights=None,                  # Heights of each plot (or single value)
				 spacing=0.05,                  # Single value or list of spacing values
				 figsize=(27, 4),               # Overall figure size
				 top_margin=0.9,                # Top margin
				 bottom_margin=0.1,             # Bottom margin
				 left_margin=0.05,              # Left margin
				 right_margin=0.05):            # Right margin
		
		self.n_plots = n_plots
		
		# Process spacing parameter - can be single value or list
		if isinstance(spacing, (int, float)):
			self.spacing = [spacing] * (n_plots - 1)  # Same spacing for all gaps
		else:
			self.spacing = spacing
			if len(spacing) != n_plots - 1:
				raise ValueError(f"Expected {n_plots-1} spacing values, got {len(spacing)}")
		
		# Set default widths if not provided
		if widths is None:
			self.widths = [1] * n_plots
		else:
			self.widths = widths
			if len(widths) != n_plots:
				raise ValueError(f"Expected {n_plots} width values, got {len(widths)}")
		
		# Handle heights - can be single value or list
		if heights is None:
			self.heights = [1] * n_plots  # Default all same height
		elif isinstance(heights, (int, float)):
			self.heights = [heights] * n_plots  # Same height for all
		else:
			self.heights = heights
			if len(heights) != n_plots:
				raise ValueError(f"Expected {n_plots} height values, got {len(heights)}")
		
		# Create figure
		self.fig = plt.figure(figsize=figsize)
		
		# Create axes directly with calculated positions
		self.axes = []
		
		# Calculate total width (sum of all relative widths)
		total_width = sum(self.widths)
		
		# Calculate total horizontal space for plots (accounting for margins)
		available_width = 1.0 - left_margin - right_margin
		
		# Calculate total spacing width in figure coordinates
		total_spacing_width = 0
		for gap_width in self.spacing:
			total_spacing_width += gap_width * available_width
		
		# Calculate actual width available for all plots
		plots_width = available_width - total_spacing_width
		
		# Calculate vertical position and standard height
		bottom = bottom_margin
		std_height = top_margin - bottom_margin
		
		# Calculate starting position
		current_x = left_margin
		
		for i in range(n_plots):
			# Calculate width of this plot in figure coordinates
			plot_width = (self.widths[i] / total_width) * plots_width
			
			# Calculate height for this plot
			plot_height = std_height * self.heights[i]
			
			# Adjust vertical position to center the plot if height is different
			y_pos = bottom + (std_height - plot_height) / 2
			
			# Create axes with exact position
			ax = self.fig.add_axes([current_x, y_pos, plot_width, plot_height])
			self.axes.append(ax)
			
			# Move to next position (including spacing if not the last plot)
			if i < n_plots - 1:
				current_x += plot_width + (self.spacing[i] * available_width)
	
		for ax in self.get_axes():
			ax.set_xticks([])
			ax.set_yticks([])

	def get_axes(self):
		"""Return all axes as a list"""
		return self.axes
	
	def show(self):
		"""Display the figure"""
		plt.show()
		
	def save(self, filename, dpi=300, **kwargs):
		"""Save the figure to a file"""
		self.fig.savefig(filename, dpi=dpi, **kwargs)