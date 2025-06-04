
from matplotlib import pyplot as plt
import numpy as np
import scipy
import matplotlib.patheffects as patheffects


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
		fig, ax = plt.subplots()

	if domain_values is None:
		domain_values = range(min(data), max(data), 1)

	y = _kde_sklearn(data, domain_values, bw) * mult
	d = scipy.zeros(len(y))
	fill_mask = y >= d

	if normalize:
		y = y/y.max()

	if fill:
		if not flip_axes:
			ax.fill_between(domain_values, y+y_offset, 0, color=color,
					 alpha=alpha, linewidth=1, zorder=zorder, ls=ls)
		else:
			ax.fill_betweenx(domain_values, y+y_offset, 0, color=color,
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
	