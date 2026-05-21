
from matplotlib import pyplot as plt
import numpy as np
import scipy
import matplotlib.patheffects as patheffects
import matplotlib.gridspec as gridspec


def plot_phase_stack(tp, prev_vec, cur_vec, name):

	if prev_vec is None:
		prev_vec = np.zeros_like(cur_vec)
	
	cur_stack = prev_vec+cur_vec
	
	plt.fill_between(tp, prev_vec, cur_stack, color=color_for_key(name), label=name)

	return cur_stack

	
def plot_H_as_growth_curve(model):

	plt.figure(figsize=(8, 2))
	plt.subplot(1, 2, 1)
	plt.imshow(model.H, aspect='auto', vmax=0.02, cmap='viridis')
	from src.plot_helpers import plot_H_as_growth_curve

	plt.subplot(1, 2, 2)
	tp = model.config.WT1_TIMEPOINTS
	H = model.H[:len(tp)]

	phase_columns = model.config.phase_columns
	keys = list(phase_columns.keys())

	plt.ylim(0, 2.)

	phases = []
	vectors = []

	for phase, indices in phase_columns.items():
		cur_vec = H[:, indices].sum(axis=1)
		vectors.append(cur_vec)
		phases.append(phase)
		
	plot_stacked_curves(tp, vectors, phases)

	plt.legend()


def plot_stacked_curves(x, vectors, names):
	
	for i in range(len(vectors)):

		if i == 0:
			previous_stack = None
	
		cur_vector = vectors[i]
		name = names[i]

		previous_stack = plot_phase_stack(x, prev_vec=previous_stack, 
										  cur_vec=cur_vector, name=name)
	
def plot_density_histogram(data, ax=None, domain_values=None, 
						  fill=False, bw=10, 
						  mult=1.0, y_offset=0, flip_axes=False, normalize=True,
						  **kwargs):
	
	from scipy.ndimage import gaussian_filter1d
	
	if ax is None:
		ax = plt.gca()
	
	if domain_values is None:
		domain_values = np.linspace(data.min(), data.max(), 200)
	
	# Create histogram then smooth it
	hist, bin_edges = np.histogram(data, bins=len(domain_values), 
								  range=(domain_values[0], domain_values[-1]))
	
	# Smooth the histogram
	smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=bw/10)
	
	# Use bin centers as x values
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
	
	y = smoothed_hist * mult
	if normalize:
		y = y/y.max()
	
	plt.plot(domain_values, y, *kwargs)

	return y

def plot_density(data, ax=None, color='red', domain_values=None, 
	alpha=1., zorder=1, fill=False, bw=10, neg=False, 
	mult=1.0, y_offset=0, flip_axes=False, lw=1, label=None, 
	ls='solid', flip_x_axis=False, normalize=True):

	from sklearn.neighbors import KernelDensity
	def _kde_sklearn(x, x_grid, bandwidth):
		kde_skl = KernelDensity(bandwidth=bandwidth)
		kde_skl.fit(x[:, np.newaxis])
		log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
		pdf = np.exp(log_pdf)
		return pdf

	if ax is None:
		ax = plt.gca()

	if domain_values is None:
		domain_values = range(min(data), max(data), 1)

	y = _kde_sklearn(data, domain_values, bw) * mult
	d = scipy.zeros(len(y))
	fill_mask = y >= d

	if normalize:
		y = y/y.max()

	if fill:
		if not flip_axes:
			ax.fill_between(domain_values, y+y_offset, y_offset, color=color,
					 alpha=alpha, linewidth=1, zorder=zorder, ls=ls)
		else:
			ax.fill_betweenx(domain_values, y+y_offset, y_offset, color=color,
					 alpha=alpha, linewidth=1, zorder=zorder, ls=ls)
	else:
		if not flip_axes:
			ax.plot(domain_values, y+y_offset, color=color,
				 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
				 solid_joinstyle='round', ls=ls)
		else:
			ax.plot(y+y_offset, domain_values, color=color,
				 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
				 solid_joinstyle='round', ls=ls)

	if not flip_axes:
		ax.set_xlim(domain_values[0], domain_values[-1])
		ax.set_ylim(0, 1.4)
	else:
		ax.set_ylim(domain_values[0], domain_values[-1])

		if flip_x_axis:
			ax.set_xlim(1.4, 0)
		else:
			ax.set_xlim(0, 1.4)

	return y

def plot_rect2(ax, x1, y1, x2, y2, color=None, facecolor=None, 
	edgecolor=None, ls='solid', alpha=1., zorder=40, lw=0.0, 
	inset=(0.0, 0.0), fill=True, joinstyle='round'):
	"""
	Plot a rectangle for ORF plotting, updated to x1 x2 and y1 y2 rather than width height
	"""

	import matplotlib.patches as patches
	from matplotlib.patches import Rectangle, FancyBboxPatch

	if edgecolor is None: edgecolor = color
	if facecolor is None: facecolor = color

	x = x1
	y = y1
	width = x2-x1
	height = y2-y1

	patch = ax.add_patch(
					patches.Rectangle(
						(x, y + inset[1]/2.0),   # (x,y)
						width, height - inset[1], # size
						facecolor=facecolor,
						edgecolor=edgecolor,
						lw=lw,
						joinstyle=joinstyle,
						ls=ls,
						fill=fill,
						alpha=alpha,
						zorder=zorder,
					))
	

def hide_spines(ax, hide_ticks=True):
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)

	if hide_ticks:
		ax.set_xticks([])
		ax.set_yticks([])



def create_sub_colormap(original_cmap_name, cmin, cmax, new_cmap_name):
	"""
	Create a new colormap based on a subrange of an existing colormap.
	
	Parameters:
	- original_cmap_name (str): Name of the original colormap.
	- cmin (float): Minimum value of the range (0 to 1).
	- cmax (float): Maximum value of the range (0 to 1).
	- new_cmap_name (str): Name for the new colormap.
	
	Returns:
	- new_cmap (LinearSegmentedColormap): New colormap based on the specified range.
	"""

	from matplotlib.colors import LinearSegmentedColormap

	
	# Get the original colormap
	original_cmap = plt.get_cmap(original_cmap_name)
	
	# Extract the colors from the original colormap within the specified range
	n_colors = 256
	original_colors = original_cmap(np.linspace(cmin, cmax, n_colors))
	
	# Create a new colormap from the extracted colors
	new_cmap = LinearSegmentedColormap.from_list(new_cmap_name, original_colors)
	
	return new_cmap


def adjust_lightness_saturation(rgba, lightness_factor, saturation_factor):
	"""
	Adjust the lightness and saturation of an RGBA color.

	:param rgba: List or tuple with four elements [r, g, b, a] where r, g, b are in range [0, 1] and a is alpha.
	:param lightness_factor: A multiplier to adjust the lightness. 1 means no change.
	:param saturation_factor: A multiplier to adjust the saturation. 1 means no change.
	:return: Adjusted RGBA color.
	"""
	import colorsys

	if len(rgba) == 3:
		r, g, b = rgba
		a = 1.
	else:
		r, g, b, a = rgba
	# Convert RGB to HLS
	h, l, s = colorsys.rgb_to_hls(r, g, b)
	
	# Adjust lightness and saturation
	l = min(max(l * lightness_factor, 0), 1)  # Ensure the new lightness is in [0, 1]
	s = min(max(s * saturation_factor, 0), 1)  # Ensure the new saturation is in [0, 1]
	
	# Convert HLS back to RGB
	r, g, b = colorsys.hls_to_rgb(h, l, s)
	
	return [r, g, b, a]

def adjust_lightness_saturation_colormap(cmap, lightness_factor, saturation_factor, new_cmap_name, N=256):
	"""
	Adjust the lightness and saturation of a Matplotlib colormap using the provided function.

	:param cmap: Matplotlib colormap (name or Colormap instance).
	:param lightness_factor: A multiplier to adjust the lightness. 1 means no change.
	:param saturation_factor: A multiplier to adjust the saturation. 1 means no change.
	:param N: Number of colors in the new colormap. Default is 256.
	:return: Adjusted colormap.
	"""
	import matplotlib.colors as mcolors
	
	if isinstance(cmap, str):
		cmap = plt.get_cmap(cmap)
	
	# Create an array to store the adjusted colors
	new_colors = []
	
	for i in np.linspace(0, 1, N):
		rgba = cmap(i)
		# Adjust the color using the provided function
		adjusted_rgba = adjust_lightness_saturation(rgba, lightness_factor, saturation_factor)
		new_colors.append(adjusted_rgba)
	
	# Create a new colormap from the adjusted colors
	new_cmap = mcolors.ListedColormap(new_colors, name=new_cmap_name)
	
	return new_cmap



def plot_heatmap_cell_cycle_tps(ax, config, plt_data, vmin, vmax, cmap,
	plot_phase_labels=True, annotations_x_offset=None, ylim_offset=None):
	from src.chromatin_model import draw_phase_label_annotations

	# Segment by g1 and s/g2m
	cg1_indices = config.get_Hpositions_for_phase('CG1')
	postG1_indices = config.get_Hpositions_for_phase('postG1')

	# define extents
	cg1_tps = config.get_phase_timepoints_for_phase('CG1')
	postG1_tps = config.get_phase_timepoints_for_phase('postG1')
	n = len(plt_data)

	g1_extent = [cg1_tps[0], cg1_tps[-1], 0, n]
	postG1_extent = [cg1_tps[-1], postG1_tps[-1], 0, n]

	cg1_plt_data = plt_data[cg1_indices]
	postg1_plt_data = plt_data[postG1_indices]

	im1 = ax.imshow(cg1_plt_data, extent=g1_extent, aspect='auto',
			   cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', interpolation='none')
	im2 = ax.imshow(postg1_plt_data, extent=postG1_extent, aspect='auto',
			   cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', interpolation='none')

	ax.set_xlim(g1_extent[0], postG1_extent[1])


	ylims = n, 0
	annotations_x_offset = n*0.034

	ylim_offset = n*0.067

	ax.set_xlabel("Cell cycle time, minutes")

	if plot_phase_labels:
		draw_phase_label_annotations(ax, config, flip=True, 
			annotations_x=n+annotations_x_offset)
		ax.set_ylim(n+ylim_offset, 0)

	return im1, im2


def color_for_key(key):
	"""Predefined colors for phases and keys for gene plots"""

	color_map = {
		 "raw": np.array([158, 50, 50])/255.,
		 "fit": np.array([145, 180, 98])/255.,
		 "R": np.array([199, 148, 144])/255.,
		 "RG1": np.array([199, 148, 144])/255.,

		 "CG1": np.array([147, 168, 198])/255.,
		 "MG1": np.array([147, 168, 198])/255.,
		 "DG1": np.array([157, 190, 201])/255.,
		 "meanG1": np.array([152, 179, 198])/255.,

		 "Delta": np.array([227, 194, 163])/255.,
		 "postG1": np.array([214, 170, 129])/255.,
		 "RpostG1": np.array([200, 170, 140])/255.,

		 "G2M": np.array([214, 170, 129])/255.,
		 "S": np.array([158, 189, 140])/255.,

		 "H": np.array([100, 100, 100])/255.
	}

	color_map['G2/M'] = color_map['G2M']

	return color_map[key]


def annotate_points(xs, ys, texts, 
				   marker='D', 
				   marker_size=20,
				   marker_color='red',
				   marker_fill='none',
				   linewidth=1,
				   text_offset=(0.05, 0.05),
				   fontsize=8,
				   text_color='black',
				   ax=None,
				   use_adjust_text=True,
				   bbox_props=None,
				   adjust_text_kwargs=None,
				   zorder=1):
	"""
	Annotate scatter plot points with custom markers and text labels.
	Uses adjustText library to prevent label overlaps when use_adjust_text=True.
	"""
	if ax is None:
		ax = plt.gca()
	
	# Plot markers
	scatter = ax.scatter(xs, ys, 
						 marker=marker, 
						 s=marker_size, 
						 edgecolor=marker_color,
						 facecolors=marker_fill,
						 linewidth=linewidth,
						 zorder=zorder)
	
	# Default parameters for adjust_text
	default_adjust_kwargs = {
		'expand_points': (1.5, 1.5),
		'force_points': (0.1, 0.2),
		'arrowprops': {'arrowstyle': '-', 'color': marker_color, 'lw': 0.5},
		'ax': ax
	}
	
	# Update with user-provided parameters
	if adjust_text_kwargs:
		default_adjust_kwargs.update(adjust_text_kwargs)
	
	# Add text labels
	text_objects = []
	
	if use_adjust_text:

		try:
			from adjustText import adjust_text
			
			# Create text objects (initially at point positions)
			for x, y, text in zip(xs, ys, texts):
				text_obj = ax.text(x, y, text,
								fontsize=fontsize,
								color=text_color,
								bbox=bbox_props,

								path_effects=[patheffects.withStroke(linewidth=2,
									foreground='white')],
								zorder=zorder)
				text_objects.append(text_obj)
			
			# Use adjustText to prevent overlaps
			adjust_text(text_objects, **default_adjust_kwargs)
			
		except ImportError:
			print("Warning: adjustText library not found. Using basic text placement instead.")
			use_adjust_text = False
	
	# Fall back to basic text placement if not using adjustText
	if not use_adjust_text:
		for x, y, text in zip(xs, ys, texts):
			offset_x, offset_y = text_offset
			text_obj = ax.annotate(text, 
							xy=(x, y),
							xytext=(x + offset_x, y + offset_y),
							fontsize=fontsize,
							color=text_color,
							bbox=bbox_props)
			text_objects.append(text_obj)
	
	return text_objects
	

def plot_deconvolution_solution(G, F, config, timepoints_wt1, timepoints_wt2, 
								plots=None, ylims=None,
								figsize=None, colors=None, line_width=3,
								data_label="Raw data", fit_label="Optimal $\\gamma$ solution"):
	"""
	Generic function to plot deconvolution solution results.
	
	Parameters:
	-----------
	plots : list, optional
		List of plot types to show. Options: ['raw', 'i', 't', 'b', 'tb']
		'raw': Data vs Fit (only available if G is not None)
		'i': Initial branch
		't': Top branch
		'b': Bottom branch  
		'tb': Combined top and bottom branches (averages t and b values and timepoints)
		Default: ['raw', 'i', 't', 'b'] if G is not None, ['i', 't', 'b'] if G is None
	"""
	import matplotlib.pyplot as plt
	import numpy as np
	
	# Set default plots
	if plots is None:
		if G is not None:
			plots = ['raw', 'i', 't', 'b']
		else:
			plots = ['i', 't', 'b']
	
	# Remove 'raw' from plots if G is None
	if G is None and 'raw' in plots:
		plots = [p for p in plots if p != 'raw']
		print("Warning: 'raw' plot removed because G is None")
	
	# Set default colors
	if colors is None:
		colors = {'data': 'black', 'fit': 'red', 'solution': 'red'}
	
	# Get indices and timepoints for different branches
	i_indices = config.get_Hpositions_for_branch('i')
	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')
	h_indices = config.get_Hpositions_for_phase('Halted')
	
	i_tps = config.get_timepoints_for_branch('i')
	t_tps = config.get_timepoints_for_branch('t')
	b_tps = config.get_timepoints_for_branch('b')
	
	# Calculate predicted values if needed
	if 'raw' in plots and G is not None:
		predicted_G = config.H @ F
	
	# Set up figure
	num_cols = len(plots)
	if figsize is None:
		figsize = (4 * num_cols, 3)
	fig, axs = plt.subplots(1, num_cols, figsize=figsize)
	
	# Ensure axs is always a list for consistent indexing
	if num_cols == 1:
		axs = [axs]


	if ylims is None:
		
		# Calculate limits
		if G is None:
			g_vmax = 0
		else:
			g_vmax = G.max()

		vmax = max(g_vmax, F.max())
		ylims = (vmax * -0.05, vmax * 1.2)

		g_ylims = (g_vmax * -0.05, g_vmax * 1.2)
		if vmax == 0:
			ylims = (-0.1, 1)
	
	# Determine split points for datasets
	n_tps1 = len(timepoints_wt1)
	n_tps2 = len(timepoints_wt2)
	
	# Plot each requested subplot
	for idx, plot_type in enumerate(plots):
		ax = axs[idx]
		
		if plot_type == 'raw':
			ax.plot(timepoints_wt1, G[:n_tps1], c=colors['data'], lw=line_width, label=data_label)
			ax.plot(timepoints_wt1, predicted_G[:n_tps1], c=colors['fit'], lw=line_width, label=fit_label)
			ax.plot(timepoints_wt2, G[n_tps1:], c=colors['data'], lw=line_width)
			ax.plot(timepoints_wt2, predicted_G[n_tps1:], c=colors['fit'], lw=line_width)
			ax.set_ylim(*g_ylims)
			ax.set_title("Data vs Fit")
			ax.legend()
			
		elif plot_type == 'i':
			ax.plot(i_tps, F[i_indices].T, c=colors['solution'], lw=line_width)
			ax.set_title("Initial branch")
			ax.set_ylim(*ylims)
			
			# Add halted line if available
			if len(h_indices) > 0:
				halted_tx = F[h_indices[0]]
				ax.axhline(halted_tx, ls='dotted', color='#555', lw=1)
				
		elif plot_type == 't':
			ax.plot(t_tps, F[t_indices].T, c=colors['solution'], lw=line_width)
			ax.set_ylim(*ylims)
			ax.set_title("Top branch")
			
		elif plot_type == 'b':
			ax.plot(b_tps, F[b_indices].T, c=colors['solution'], lw=line_width)
			ax.set_ylim(*ylims)
			ax.set_title("Bottom branch")
			
		elif plot_type == 'tb':
			# Combined top and bottom branches
			# Average the F values and timepoints
			if len(t_tps) != len(b_tps):
				raise ValueError("Top and bottom branches must have the same number of timepoints for combined plot")
			if len(t_indices) != len(b_indices):
				raise ValueError("Top and bottom branches must have the same number of indices for combined plot")
			
			combined_F = (F[t_indices] + F[b_indices]) / 2
			combined_tps = (np.array(t_tps) + np.array(b_tps)) / 2
			
			ax.plot(combined_tps, combined_F.T, c=colors['solution'], lw=line_width)
			ax.set_ylim(*ylims)
			ax.set_title("Combined Top/Bottom branches")
			
		else:
			raise ValueError(f"Unknown plot type: {plot_type}. Valid options are: 'raw', 'i', 't', 'b', 'tb'")
	
	plt.tight_layout()
	return fig, axs


def create_subplot_pairs(pair_rows, pair_cols, pair_spacing=0.3, hspacing=0.3, figsize=(12, 8), 
						within_pair_spacing=0, figure_spacing=None):
	"""

	Useful for plotting raw data replicates and deconvolved data (recovery and mother/daughter).

	The raw data replicates are two plots separated by a smaller difference than the deconvolved data.

	See the Figure Loci example for example usage.
	
	Parameters:
	-----------
	pair_rows : int
		Number of rows of subplot pairs
	pair_cols : int  
		Number of columns of subplot pairs
	pair_spacing : float, default 0.3
		Spacing between pairs (both wspace and hspace)
	figsize : tuple, default (12, 8)
		Figure size as (width, height)
	within_pair_spacing : float, default 0
		Spacing within each pair
	figure_spacing : dict, optional
		Dict with keys 'left', 'right', 'top', 'bottom' for figure margins
		
	Returns:
	--------
	fig : matplotlib.figure.Figure
		The figure object
	axes_pairs : list of tuples
		List of (left_ax, right_ax) tuples, one for each pair
		Ordered row-wise: [(row0_col0_pair), (row0_col1_pair), ...]
	"""

	import matplotlib.gridspec as gridspec

	# Create figure
	fig = plt.figure(figsize=figsize)
	
	# Set figure spacing if provided
	if figure_spacing:
		plt.subplots_adjust(**figure_spacing)
	
	# Create main GridSpec for positioning pairs
	main_gs = gridspec.GridSpec(pair_rows, pair_cols, 
							   wspace=pair_spacing, 
							   hspace=hspacing, top=0.8)
	
	axes_pairs = []
	
	# Create nested GridSpecs for each pair
	for row in range(pair_rows):
		for col in range(pair_cols):
			# Create nested GridSpec within this cell (1 row, 2 columns)
			nested_gs = gridspec.GridSpecFromSubplotSpec(
				1, 2, main_gs[row, col], 
				wspace=within_pair_spacing
			)
			
			# Create the two subplots for this pair
			left_ax = fig.add_subplot(nested_gs[0, 0])
			right_ax = fig.add_subplot(nested_gs[0, 1])
			
			axes_pairs.append((left_ax, right_ax))
	
	return fig, axes_pairs

def add_pair_title(fig, pair_axes, title, **text_kwargs):
	"""Specifically for the pair subplots defined in
	create_subplot_pairs"""

	"""Add a title centered above a pair of axes"""
	left_ax, right_ax = pair_axes
	
	# Get positions of both axes
	left_pos = left_ax.get_position()
	right_pos = right_ax.get_position()
	
	# Calculate center x and top y
	center_x = (left_pos.x0 + right_pos.x1) / 2
	top_y = max(left_pos.y1, right_pos.y1) + 0.02  # Small offset above
	
	fig.text(center_x, top_y, title, ha='center', va='bottom', **text_kwargs)


def create_three_subplot_layout(figsize=(12, 4), left_padding=0.1, right_padding=0.1, 
							   top_padding=0.1, bottom_padding=0.1, 
							   wspace_left=0.3, wspace_right=0.4):
	"""
	Create a figure with three subplots arranged as: [subplot1] [subplot2] | [subplot3]
	
	Parameters:
	-----------
	figsize : tuple, default (12, 4)
		Figure size (width, height) in inches
	left_padding : float, default 0.1
		Left margin of the figure
	right_padding : float, default 0.1
		Right margin of the figure
	top_padding : float, default 0.1
		Top margin of the figure
	bottom_padding : float, default 0.1
		Bottom margin of the figure
	wspace_left : float, default 0.3
		Horizontal spacing between subplot1 and subplot2
	wspace_right : float, default 0.4
		Horizontal spacing between subplot2 and subplot3
	
	Returns:
	--------
	fig : matplotlib.figure.Figure
		The figure object
	ax1 : matplotlib.axes.Axes
		First subplot (left)
	ax2 : matplotlib.axes.Axes
		Second subplot (middle)
	ax3 : matplotlib.axes.Axes
		Third subplot (right)
	"""
	
	# Create figure
	fig = plt.figure(figsize=figsize)
	
	# Create a gridspec with 1 row and 4 columns
	# We'll use 4 columns to have more control over spacing
	gs = gridspec.GridSpec(1, 4, 
						  left=left_padding, 
						  right=1-right_padding,
						  top=1-top_padding, 
						  bottom=bottom_padding,
						  wspace=0)  # Set wspace to 0, we'll handle spacing manually
	
	# Calculate relative widths for the subplots
	# subplot1 and subplot2 should be equal width
	# subplot3 can be a different width
	subplot_width = 1.0  # Base width unit
	
	# Create subplots with manual spacing
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1])  
	ax3 = fig.add_subplot(gs[0, 3])  # Skip column 2 for spacing
	
	# Adjust subplot positions manually for better control
	pos1 = ax1.get_position()
	pos2 = ax2.get_position()
	pos3 = ax3.get_position()
	
	# Calculate new positions with desired spacing
	total_width = 1 - left_padding - right_padding
	subplot_width = (total_width - wspace_left - wspace_right) / 3
	
	# Reposition subplots
	ax1.set_position([left_padding, bottom_padding, 
					 subplot_width, 1 - top_padding - bottom_padding])
	
	ax2.set_position([left_padding + subplot_width + wspace_left, bottom_padding,
					 subplot_width, 1 - top_padding - bottom_padding])
	
	ax3.set_position([left_padding + 2*subplot_width + wspace_left + wspace_right, bottom_padding,
					 subplot_width, 1 - top_padding - bottom_padding])
	
	return fig, ax1, ax2, ax3


def create_nine_subplot_layout(figsize=(8, 3), left_padding=0.08, right_padding=0.02, 
							  top_padding=0.16, bottom_padding=0.08, 
							  hspace=0.03, wspace_col1_col2=0.0, wspace_col2_col3=0.05):
	"""
	Create a figure with nine subplots arranged in a 3x3 grid with custom column spacing.
	
	Parameters:
	-----------
	figsize : tuple, default (12, 9)
		Figure size (width, height) in inches
	left_padding : float, default 0.08
		Left margin of the figure
	right_padding : float, default 0.02
		Right margin of the figure
	top_padding : float, default 0.08
		Top margin of the figure
	bottom_padding : float, default 0.08
		Bottom margin of the figure
	hspace : float, default 0.3
		Vertical spacing between subplot rows
	wspace_col1_col2 : float, default 0.0
		Horizontal spacing between column 1 and column 2
	wspace_col2_col3 : float, default 0.1
		Horizontal spacing between column 2 and column 3
	
	Returns:
	--------
	fig : matplotlib.figure.Figure
		The figure object
	axes : list of list
		3x3 array of axes objects, axes[row][col]
	"""
	
	# Create figure
	fig = plt.figure(figsize=figsize)
	
	# Calculate dimensions
	plot_width = 1 - left_padding - right_padding
	plot_height = 1 - top_padding - bottom_padding
	
	# Calculate column widths and positions
	total_col_spacing = wspace_col1_col2 + wspace_col2_col3
	subplot_width = (plot_width - total_col_spacing) / 3
	subplot_height = (plot_height - 2 * hspace) / 3
	
	# Column x-positions
	col_x_positions = [
		left_padding,  # Column 1
		left_padding + subplot_width + wspace_col1_col2,  # Column 2
		left_padding + 2 * subplot_width + wspace_col1_col2 + wspace_col2_col3  # Column 3
	]
	
	# Row y-positions (from top to bottom)
	row_y_positions = [
		1 - top_padding - subplot_height,  # Row 0 (top)
		1 - top_padding - 2 * subplot_height - hspace,  # Row 1 (middle)
		1 - top_padding - 3 * subplot_height - 2 * hspace  # Row 2 (bottom)
	]
	
	# Create all 9 subplots with manual positioning
	axes = []
	for row in range(3):
		axes_row = []
		for col in range(3):
			ax = fig.add_axes([col_x_positions[col], row_y_positions[row], 
							  subplot_width, subplot_height])
			axes_row.append(ax)
		axes.append(axes_row)
	
	return fig, axes


def add_trajectory_arrows(ax, x_data, y_data, index, index_offset, arrow_scale=0.1, 
						 arrow_color='red', arrow_alpha=1.0):
	"""
	Add short directional arrows along a trajectory path.
	"""
	from matplotlib.patches import FancyArrowPatch
	from matplotlib.patches import FancyArrow

	# Plot an area of the trajectory of the start of the path
	dx =  x_data[index+index_offset] - x_data[index]
	dy = y_data[index+index_offset] - y_data[index]
	
	# ---------------- Aspect ratio correction ---------------------

	# Calculate effective aspect ratio (data range * physical ratio)
	# Calculate the data aspect ratio
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	x_range = xlim[1] - xlim[0]
	y_range = ylim[1] - ylim[0]
	fig = ax.get_figure()
	fig_width, fig_height = fig.get_size_inches()
	aspect_ratio = (x_range / y_range) * (fig_height / fig_width)

	# Adjust arrow dimensions based on aspect ratio and data ranges
	# Scale dimensions as fractions of the respective axis ranges
	head_width = 0.2
	head_length = 0.2
	body_width = 0.01

	corrected_head_width = head_width * y_range
	corrected_head_length = head_length * x_range
	corrected_body_width = body_width * y_range
	
	# Apply aspect correction to keep proportions correct
	# If plot is wider than tall, adjust width-related parameters
	if aspect_ratio < 1:  # Plot is wider than tall
		corrected_head_width *= aspect_ratio
		corrected_body_width *= aspect_ratio
	else:  # Plot is taller than wide
		corrected_head_length /= aspect_ratio

	# ---------------- Aspect ratio correction ---------------------

	# Normalize and scale the arrow
	length = np.sqrt(dx**2 + dy**2)
	if length > 0:
		dx_norm = (dx / length) * arrow_scale
		dy_norm = (dy / length) * arrow_scale
	
		arrow = FancyArrow(
					x_data[index], y_data[index], 
					dx_norm, dy_norm,
					width=corrected_body_width,
					head_width=corrected_head_width,
					head_length=corrected_head_length,
					overhang=0.,
					color=arrow_color,
					alpha=arrow_alpha
				)

		ax.add_patch(arrow)
		

# For the last chromatin axes, plot the genomic scale legend
def add_im_genomic_scale_legend(ax, x_start, x_length, legend_y=-80):
	import matplotlib.patheffects as path_effects
	ax.plot([x_start, x_start+x_length], [legend_y, legend_y], 
		linewidth=3, color='gray',
			clip_on=False, label='Scale line (2 units)')
	ax.text((x_start+x_start+x_length)/2., legend_y, f"{x_length} bp", 
		   va='center', ha='center',
		   path_effects=[path_effects.withStroke(linewidth=3, 
												foreground='white')])
			

def _plot_index_label(ax, x, y, index, total):
	# Add label for index in total of the branch
	ax.text(x, y, f"{index} of {total}", clip_on=False)



def get_truncated_RdBu_r():
	from matplotlib.colors import LinearSegmentedColormap
	
	# Get the original colormap
	original_cmap = plt.get_cmap('RdBu_r')
	new_cmap_name = 'truncated_RdBu_r'
	cmin = 0.25
	cmax = 0.75

	# Extract the colors from the original colormap within the specified range
	n_colors = 256
	original_colors = original_cmap(np.linspace(cmin, cmax, n_colors))

	# Create a new colormap from the extracted colors
	new_cmap = LinearSegmentedColormap.from_list(new_cmap_name, original_colors)
	return new_cmap


def create_proportional_subplots_vertical(sizes, labels=None, figsize=(8, 12), 
										vertical_padding=0.15,
										subplot_kw=None, **fig_kwargs):
	"""
	Create vertical subplots with heights proportional to the given sizes using gridspec.
	
	Used for plotting histone modifications grouped by modification type, where each
	group gets a subplot with height proportional to the number of modifications.
	
	Args:
		sizes (list): List of sizes/counts that determine the relative heights
		labels (list, optional): Labels for each subplot. If None, uses indices.
		figsize (tuple): Figure size (width, height)
		vertical_padding (float): Vertical spacing between rows (hspace)
		subplot_kw (dict, optional): Keyword arguments to pass to subplot creation
		**fig_kwargs: Additional keyword arguments for figure creation
		
	Returns:
		tuple: (fig, axes) where axes is a list of matplotlib axes objects
	"""
	if not sizes or all(s == 0 for s in sizes):
		raise ValueError("All sizes cannot be zero or empty")
	
	# Convert sizes to height ratios (normalize to avoid very small/large numbers)
	total_size = sum(sizes)
	height_ratios = [size / total_size * 100 for size in sizes]  # Scale to reasonable numbers
	
	# Create figure and gridspec for vertical layout
	fig = plt.figure(figsize=figsize, **fig_kwargs)
	gs = gridspec.GridSpec(len(sizes), 1, 
						  height_ratios=height_ratios,
						  hspace=vertical_padding)
	
	# Create subplots
	axes = []
	subplot_kw = subplot_kw or {}
	
	for row in range(len(sizes)):
		ax = fig.add_subplot(gs[row, 0], **subplot_kw)
		axes.append(ax)
		
		# Add title if labels provided
		if labels and row < len(labels):
			ax.set_title(f"{labels[row]}", fontsize=14, pad=8)
	
	return fig, axes

def create_proportional_subplots(sizes, n_rows, labels=None, figsize=(15, 18), 
								  horizontal_padding=0.3, vertical_padding=0.2,
								  title_rows=[0], height_ratios=None,
								  subplot_kw=None, **fig_kwargs):
	"""
	Create n rows of subplots with widths proportional to the given sizes using gridspec.

	Generic version that allows for any number of rows.
	
	Args:
		sizes (list): List of sizes/counts that determine the relative widths
		n_rows (int): Number of rows to create
		labels (list, optional): Labels for each subplot. If None, uses indices.
		figsize (tuple): Figure size (width, height)
		horizontal_padding (float): Horizontal spacing between columns (wspace)
		vertical_padding (float): Vertical spacing between rows (hspace)
		title_rows (list): List of row indices that should have titles
		height_ratios (list, optional): Custom height ratios for rows. If None, uses equal heights.
		subplot_kw (dict, optional): Keyword arguments to pass to subplot creation
		**fig_kwargs: Additional keyword arguments for figure creation
		
	Returns:
		tuple: (fig, axes) where axes is a 2D list [row][col] of matplotlib axes objects
	"""
	if not sizes or all(s == 0 for s in sizes):
		raise ValueError("All sizes cannot be zero or empty")
	
	if n_rows <= 0:
		raise ValueError("Number of rows must be positive")
	
	# Convert sizes to width ratios (normalize to avoid very small/large numbers)
	total_size = sum(sizes)
	width_ratios = [size / total_size * 100 for size in sizes]  # Scale to reasonable numbers
	
	# Set up height ratios - default to equal heights if not specified
	if height_ratios is None:
		height_ratios = [1] * n_rows
	elif len(height_ratios) != n_rows:
		raise ValueError(f"height_ratios length ({len(height_ratios)}) must match n_rows ({n_rows})")
	
	# Create figure and gridspec for n rows
	fig = plt.figure(figsize=figsize, **fig_kwargs)
	gs = gridspec.GridSpec(n_rows, len(sizes), 
						  width_ratios=width_ratios,
						  height_ratios=height_ratios,
						  hspace=vertical_padding, 
						  wspace=horizontal_padding)
	
	# Create subplots - 2D structure [row][col]
	axes = []
	subplot_kw = subplot_kw or {}
	
	for row in range(n_rows):
		row_axes = []
		for col in range(len(sizes)):
			ax = fig.add_subplot(gs[row, col], **subplot_kw)
			row_axes.append(ax)
			ax.set_xticks([])
			ax.set_yticks([])            
			
			# Add title to rows where needed
			if row in title_rows:
				if labels and col < len(labels):
					ax.set_title(f"{labels[col]}", fontsize=18, pad=8)
		
		axes.append(row_axes)
	
	return fig, axes


def create_proportional_subplots_3rows(sizes, labels=None, figsize=(15, 18), 
									   horizontal_padding=0.3, vertical_padding=0.2,
									   title_rows=[0],
									   subplot_kw=None, **fig_kwargs):
	"""
	Create 3 rows of subplots with widths proportional to the given sizes using gridspec.

	Used in nucleosome histones heatmap plotting to group histone modifications.
	See fig_nucleosomes.py and histones.py
	
	Args:
		sizes (list): List of sizes/counts that determine the relative widths
		labels (list, optional): Labels for each subplot. If None, uses indices.
		figsize (tuple): Figure size (width, height)
		horizontal_padding (float): Horizontal spacing between columns (wspace)
		vertical_padding (float): Vertical spacing between rows (hspace)
		title_rows (list): List of row indices that should have titles
		subplot_kw (dict, optional): Keyword arguments to pass to subplot creation
		**fig_kwargs: Additional keyword arguments for figure creation
		
	Returns:
		tuple: (fig, axes) where axes is a 2D list [row][col] of matplotlib axes objects
	"""
	print(sizes)  # Keep your existing print statement
	
	# Call the generic function with 3 rows
	return create_proportional_subplots(
		sizes=sizes, 
		n_rows=3,
		labels=labels, 
		figsize=figsize,
		horizontal_padding=horizontal_padding,
		vertical_padding=vertical_padding,
		title_rows=title_rows,
		height_ratios=None,  # Uses equal heights [1, 1, 1]
		subplot_kw=subplot_kw,
		**fig_kwargs
	)

def blend_colors(color1, color2, alpha=0.5):
	"""Blend two colors together"""
	# Convert colors to RGB if they're named colors or hex
	import matplotlib.colors as mcolors

	rgb1 = mcolors.to_rgb(color1)
	rgb2 = mcolors.to_rgb(color2)
	
	# Blend the colors
	blended = tuple(alpha * c1 + (1 - alpha) * c2 for c1, c2 in zip(rgb1, rgb2))
	return blended

def blend_three_colors(color1, color2, color3):
	"""Blend three colors together"""
	import matplotlib.colors as mcolors
	rgb1 = mcolors.to_rgb(color1)
	rgb2 = mcolors.to_rgb(color2) 
	rgb3 = mcolors.to_rgb(color3)
	
	# Average the three colors
	blended = tuple((c1 + c2 + c3) / 3 for c1, c2, c3 in zip(rgb1, rgb2, rgb3))
	return blended


def plot_trajectory_deconvolved_values(config, 
	x_values, y_values, 
	x_key=None, y_key='Expression', xlim=(-0.5, 8), ylim=(-0.5, 12), 
	lw=5, plot_arrows=True, color_arrows=True, ax=None,
	arrow_trajectory_offset=0.0, values_are_preindexed_by_branch=False):
	"""
	Plot chromatin vs expression data colored by cell cycle phase for a single gene.
	"""
	
	# Plot by cell cycle phase
	from src.plot_helpers import color_for_key
	phases = ['G2M', 'S', 'meanG1']

	# Set the limits automatically
	from src.math_utils import create_data_lims

	if xlim is None:
		xlim = create_data_lims(x_values, padding=0.2)

	if ylim is None:
		ylim = create_data_lims(y_values, padding=0.2)

	# plt.xlim(*xlim)
	# plt.ylim(*ylim)

	if ax is None:
		ax = plt.gca()  # Get current axes

	# Select indices values for plotting, mean t and b
	if not values_are_preindexed_by_branch:
		t_indices = config.t_indices()
		b_indices = config.b_indices()

		t_x_values = x_values[t_indices].values
		b_x_values = x_values[b_indices].values
		x_values = (t_x_values+b_x_values)/2

		t_y_values = y_values[t_indices].values
		b_y_values = y_values[b_indices].values
		y_values = (t_y_values+b_y_values)/2

	# Repeat the last value to close the loop
	looped_y_values = np.concatenate([y_values, y_values[0:1]])
	looped_x_values = np.concatenate([x_values, x_values[0:1]])

	if plot_arrows:
		# def _create_offset_curve(x_loop, y_loop, offset):
		# 	from src.math_utils import compute_offset_curve, offset_curve_shapely

		# 	x_loop_offset, y_loop_offset = offset_curve_shapely(x_loop, y_loop, 
		# 		offset)
			# return x_loop_offset, y_loop_offset

		# x_loop_offset, y_loop_offset = _create_offset_curve(looped_x_values, 
		# 	looped_y_values, offset=arrow_trajectory_offset)

		x_loop_offset, y_loop_offset = looped_x_values, looped_y_values

		from matplotlib.patches import FancyArrowPatch
		from src.math_utils import compute_signed_area

		signed_area = compute_signed_area(looped_x_values, looped_y_values)
		is_ccw = signed_area > 0

		# Color arrows by CCW or CW
		if is_ccw:
			ls = 'solid'
			start_marker_color = 'black'
		else:
			ls = (0, (2, 1))
			start_marker_color = 'white'

		if abs(signed_area) > 0.00:

			arrow_index_offset = 0

			# Find the index in which the trajectory has traveled
			# some distance
			def _compute_dist_traveled(x1, y1, index):
				return (x2-x1)**2 + (y2-y1)**2

			def _cumulative_distance(x, y):
				"""
				Compute cumulative distance traveled from arrays of x and y coordinates.
				"""
				x = np.array(x)
				y = np.array(y)
				
				# Compute differences between consecutive points
				dx = np.diff(x)
				dy = np.diff(y)
				
				# Compute distance between consecutive points using Pythagorean theorem
				distances = np.sqrt(dx**2 + dy**2)
				
				# Compute cumulative sum, prepending 0 for the starting point
				cumulative_dist = np.concatenate([[0], np.cumsum(distances)])

				return cumulative_dist

			min_proportion_traveled = 0.20
			trajectory_distances = _cumulative_distance(x_loop_offset, y_loop_offset)
			total_distance = trajectory_distances[-1]
			arrow_index_offset = np.argmax(trajectory_distances >= min_proportion_traveled\
				 * trajectory_distances[-1])

			ax.plot(x_loop_offset[:arrow_index_offset], 
				y_loop_offset[:arrow_index_offset], c='black',
			ls=ls, lw=3)

			# Add arrowhead
			arrow = FancyArrowPatch((x_loop_offset[arrow_index_offset-1], 
									 y_loop_offset[arrow_index_offset-1]), 
									(x_loop_offset[arrow_index_offset+1], 
									 y_loop_offset[arrow_index_offset+1]),
									arrowstyle='-|>', mutation_scale=16, 
									color='black', lw=0)

			ax.add_patch(arrow)
			ax.scatter(x_loop_offset[0],
				y_loop_offset[0], s=40, marker='D', facecolor=start_marker_color,
				edgecolor='black', zorder=10, lw=2)


	for phase in phases:

		# Use absolute indexing, that is the first indices represent
		# the mean g1, then s, then G2M
		g1_indices = range(0, len(config.get_Hpositions_for_phase('CG1')))
		s_indices = range(g1_indices[-1], g1_indices[-1]+
			len(config.get_Hpositions_for_phase('S')))
		g2m_indices = list(range(s_indices[-1], s_indices[-1]+
			len(config.get_Hpositions_for_phase('G2M')))) + [0] # encircle

		phase_indices_map = {
			'meanG1': g1_indices,
			'S': s_indices,
			'G2M': g2m_indices,
		}

		cur_y_values = looped_y_values[phase_indices_map[phase]]
		cur_x_values = looped_x_values[phase_indices_map[phase]]

		color = color_for_key(phase)
		ax.plot(cur_x_values, cur_y_values,
				   lw=lw, color=color, zorder=0)

		# if phase == 'meanG1':
		# Add start position
		# ax.scatter(cur_x_values[0], cur_y_values[0],
		# 		   s=10, color='black', marker='D', label=phase, zorder=1)

	plt.xlim(*xlim)
	plt.ylim(*ylim)


def create_first_character_title(param_str):
	"""Capitalize the first character in a string. Useful for title casing
	for figures."""
	return param_str[0].upper() + param_str[1:]


def plot_composite_heatmap(average_composite_data, title, num_origins, extent=[-2000, 2000, 0, 260],
	xlims=(-1000, 1000), figsize=(5, 5), show_footprint_box=False, footprint_box=(0, 50, 40, 120)):
	"""
	Plot average oriented MNase heatmap across a set of origins.
	
	Parameters
	----------
	average_composite_data : np.ndarray
		Averaged 3D MNase data (time x fragment x position)
	title : str
		Plot title
	num_origins : int
		Number of origins in the composite, shown in title
	"""
	from src.config import load_default_chrom_configs

	config1, _ = load_default_chrom_configs()
	t_indices = config1.t_indices()
	b_indices = config1.b_indices()
	num_rows = 8
	fig, axs = plt.subplots(num_rows, 1, figsize=figsize)
	step = len(t_indices) // num_rows
	extent = extent

	for i in range(0, num_rows):
		ax = axs[i]
		plot_index_t = t_indices[i * step]
		plot_index_b = b_indices[i * step]
		composite_data = (average_composite_data[plot_index_t] +
						  average_composite_data[plot_index_b]) / 2.

		ax.imshow(composite_data, cmap='magma_r',
				  origin='lower', aspect='auto', vmin=0, vmax=15,
				  extent=extent)

		if i == 1:
			ax.set_yticks([0])
			ax.set_yticklabels(['Mean G1'], rotation=90, ha='right', va='center')
		elif i == 4:
			ax.set_yticks([0])
			ax.set_yticklabels(['S'], rotation=90, ha='right', va='center')
		elif i == 6:
			ax.set_yticks([0])
			ax.set_yticklabels(['G2/M'], rotation=90, ha='right', va='center')
		else:
			ax.set_yticks([])

		if i == 0:
			ax.set_yticks([260], minor=True)
		elif i in [3, 5, 7]:
			ax.set_yticks([0], minor=True)

		if i == num_rows - 1:
			ax.set_xticks(np.arange(extent[0], extent[1], 500))
			ax.set_xlabel("Position from origin center, bp")
		else:
			ax.set_xticks([])

		ax.tick_params(axis='y', which='major', length=0, pad=2)
		ax.tick_params(axis='y', which='minor', length=13)
		ax.set_xlim(*xlims)

		if show_footprint_box:
			x1, x2, y1, y2 = footprint_box
			plot_rect2(ax,
				x1=x1, x2=x2, y1=y1, y2=y2,
				edgecolor='blue', facecolor='none',
				lw=0.8, alpha=0.4, zorder=50, fill=False)

	plt.subplots_adjust(hspace=0)
	plt.suptitle(f"{title}, n={num_origins}", fontweight='demi', fontsize=16)
