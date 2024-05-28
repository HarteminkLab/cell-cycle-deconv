
from matplotlib import pyplot as plt
import numpy as np
import scipy


def plot_phase_stack(tp, prev_vec, cur_vec, name):
	from src.model import color_for_key

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
	
def plot_density(data, ax=None, color='red', arange=None, 
	alpha=1., zorder=1, fill=False, bw=10, neg=False, 
	mult=1.0, y_offset=0, flip=False, lw=1, label=None, ls='solid'):

	from sklearn.neighbors import KernelDensity
	def _kde_sklearn(x, x_grid, bandwidth):
		kde_skl = KernelDensity(bandwidth=bandwidth)
		kde_skl.fit(x[:, np.newaxis])
		log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
		pdf = np.exp(log_pdf)
		return pdf

	if ax is None:
		fig, ax = plt.subplots()

	if arange is None:
		arange = min(data), max(data), 1

	x = np.arange(arange[0], arange[1], arange[2])

	y = _kde_sklearn(data, x, bw) * mult
	d = scipy.zeros(len(y))
	fill_mask = y >= d

	if fill:
		if not flip:
			ax.fill_between(x, y+y_offset, 0, color=color,
					 alpha=alpha, linewidth=1, zorder=zorder, ls=ls)
		else:
			ax.fill_betweenx(x, y+y_offset, 0, color=color,
					 alpha=alpha, linewidth=1, zorder=zorder, ls=ls)
	else:
		if not flip:
			ax.plot(x, y+y_offset, color=color,
				 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
				 solid_joinstyle='round', ls=ls)
		else:
			ax.plot(y+y_offset, x, color=color,
				 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
				 solid_joinstyle='round', ls=ls)

	return y

def plot_rect2(ax, x1, y1, x2, y2, color=None, facecolor=None, 
	edgecolor=None, ls='solid', fill_alpha=1., zorder=40, lw=0.0, 
	inset=(0.0, 0.0), fill=True, joinstyle='round'):
	"""
	Plot a rectangle for ORF plotting, updated to x1 x2 and y1 y2 rather than width height
	"""

	import matplotlib.patches as patches

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
						alpha=fill_alpha,
						zorder=zorder
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