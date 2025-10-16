
import numpy as np
from matplotlib import pyplot as plt


def _compute_entropy_3d(loaded_data, mode='position'):
	"""Compute the entropy for a 3d matrix. Entropy can be 
	computed along the second and third dimensions. Typically,
	we will compute this along the positional dimension to 
	capture nucleosome positional variation.

	Dimensions: 
		1. Time points
		2. Fragment
		3. Position
	"""
	from src.helpers import calc_entropy

	origin_center_region = loaded_data

	eps = 1e-5
	def full_entropy(matrix):
		# Use all fragment and positions in calculation
		return calc_entropy(matrix.flatten()+eps)
	
	def frag_entropy(matrix):
		# Summarize the positional dimension
		return calc_entropy(matrix.mean(1)+eps)
	
	def positional_entropy(matrix):
		# Summarize the fragment dimension
		return calc_entropy(matrix.mean(0)+eps)
	
	if mode == 'full':
		entropy_func = full_entropy
	elif mode == 'fragment':
		entropy_func = frag_entropy
	elif mode == 'position':
		entropy_func = positional_entropy

	entropy_values = [entropy_func(origin_center_region[i, :, :]) \
					  for i in range(origin_center_region.shape[0])]
	
	return np.array(entropy_values)



def fold_halves_together(dat, axis=1):
	# Assumes that dat is an odd dimension on the folding axes
	# That is the center of the data is on a position of interest
	
	if axis == 1:
		half_ind = (dat.shape[1]-1)//2
		first_half = dat[:, :half_ind]
		second_half = dat[:, half_ind+1:]
		first_half_flipped = np.flip(first_half, axis=1)

		# Add back the center position
		center_values = dat[:, half_ind:half_ind+1]
		combined = (first_half_flipped + second_half)/2
		combined = np.concatenate([center_values, combined], axis=axis)
	else:
		combined = fold_halves_together(dat.T, axis=1)
		combined = combined.T

	return combined


def select_region_away(data_loader, pos, distance_away, window=1000,
					  absolute=False):
	from src.global_config import GlobalConstants
	
	window_2 = window//2
	nucleosome_fragments = 130, 200

	bin_width, bin_height = GlobalConstants.BIN_WIDTH, GlobalConstants.BIN_HEIGHT
	
	selected_region = pos-window_2+distance_away, \
		pos+window_2+distance_away
	
	offset = data_loader.loaded_subset_span[0]
	selected_indices = (selected_region[0]-offset)//GlobalConstants.BIN_WIDTH,\
		(selected_region[1]-offset)//GlobalConstants.BIN_WIDTH
	selected_fragment_indices = nucleosome_fragments[0]//GlobalConstants.BIN_HEIGHT,\
		nucleosome_fragments[1]//GlobalConstants.BIN_HEIGHT

	subset_data = data_loader.loaded_subset_data[:, 
		selected_fragment_indices[0]:selected_fragment_indices[1], 
		selected_indices[0]:selected_indices[1]]
		
	return subset_data

def _plot_peaks(tb_data, x_values, flipped=False):

	if flipped:
		xs = np.arange(tb_data.shape[1])
		ys = tb_data.argmax(0)
	else:
		ys = np.arange(tb_data.shape[0])
		xs = tb_data.argmax(1)

	def _compute_lowess(x, y, frac=0.2):
		from statsmodels.nonparametric.smoothers_lowess import lowess

		if flipped:
			smoothed = lowess(x, y, frac=frac)
			x, y = x_values, smoothed[:, 1]
		else:
			smoothed = lowess(x_values, y, frac=frac)
			x, y = smoothed[:, 1], smoothed[:, 0]

		return x, y

	lo_x, lo_y = _compute_lowess(ys, xs)

	plt.scatter(x_values, ys, s=1, c='black', alpha=0.5)
	plt.plot(lo_x, lo_y, color='black', lw=1, alpha=0.75)

def _normed_data(b_indices_dat):
	b_indices_dat = (b_indices_dat - b_indices_dat.mean(1)[:, None]) / \
		(b_indices_dat.std(1)[:, None]+1e-5)
	return b_indices_dat


def generate_origin_entropy_matrix(origins, data_loader,
	furthest_dist=20000, step=1000, num_timepoints=257, 
	window=2000,
	position_key='pos'):
	from src.timer import Timer

	timer = Timer()

	distances_away = list(range(-furthest_dist, furthest_dist+step, step))

	num_selected_origins = len(origins)

	origin_entropies_matrix = np.zeros((num_selected_origins, len(distances_away), 
												 num_timepoints))

	for j, (origin_id, origin) in enumerate(origins.iterrows()):

		load_span = origin[position_key] - furthest_dist, origin[position_key] + furthest_dist
		loaded_data, loaded_span = data_loader.load_mnase_span(origin.chr, load_span)

		origin_entropy_distances = np.zeros((len(distances_away), num_timepoints))

		for i, dist in enumerate(distances_away):
			subset_region = select_region_away(data_loader, origin[position_key], dist,
											window=window, absolute=False)
			entropy_values = _compute_entropy_3d(subset_region)
			origin_entropy_distances[i] = entropy_values

		normed_values = (origin_entropy_distances - origin_entropy_distances.mean(1)[:, None])/\
			(origin_entropy_distances.std(1)[:, None]+1e-5)

		origin_entropies_matrix[j] = normed_values

		if j % 10 == 0:
			timer.print_time(f"{j}/{len(origins)}")

	return origin_entropies_matrix


def plot_entropy_by_distance(deconv_all_origin_dist_entropies, config1,
	folded=True, furthest_dist=20000, step=1000, title=None, xlabel="Distance from origin"):

	vmax = 1.75

	plt.figure(figsize=(4.5, 5))

	b_indices_dat = deconv_all_origin_dist_entropies[:, :, config1.b_indices()]
	t_indices_dat = deconv_all_origin_dist_entropies[:, :, config1.t_indices()]
	num_origins = len(deconv_all_origin_dist_entropies)

	tb_data = (b_indices_dat+t_indices_dat)/2.

	from src.config import retrieve_phase_index_ticks
	phase_ticks, edge_ticks, tick_labels = retrieve_phase_index_ticks('t', config1, with_labels=True)

	ax = plt.gca()
	ax.set_yticks(phase_ticks)
	ax.set_yticklabels(tick_labels, fontsize=8, rotation=90, ha='right', va='center')
	ax.set_yticks(edge_ticks, minor=True)

	ax.tick_params(axis='y', which='major', length=0)
	ax.tick_params(axis='y', which='minor', length=10) 

	step_2 = step//2
	tick_step = 10000
	tick_start = (furthest_dist//5000)*5000
	xticks = np.arange(-tick_start, tick_start+tick_step, tick_step)
	xtick_labels = [f"{x/step:.0f} kb" if x <= 0 else 
		f"+{x/step:.0f} kb" for x in xticks]
	plt.xticks(xticks, xtick_labels)

	cmap = 'RdBu_r'

	if folded:
		deconv_folded_data = fold_halves_together(tb_data)
		plot_data_folded = _normed_data(deconv_folded_data.mean(0)).T

		x_values = np.arange(0, furthest_dist+step, step)
		extent = [-step_2, furthest_dist+step_2, 128, 0]

		im = plt.imshow(plot_data_folded, cmap=cmap,
		        aspect='auto', vmin=-vmax, vmax=vmax, 
		        extent=extent,
		           interpolation='none')
		cbar = plt.colorbar()

		_plot_peaks(plot_data_folded, x_values, flipped=True)
		plt.title("Absolute distance from origin")

		plt.xlim(0, 25000)
		cbar.ax.set_ylabel("Normalized entropy", rotation=270, va='bottom')
		plt.xlabel(xlabel)

	else:
		plot_data = _normed_data(tb_data.mean(0)).T

		extent = [-furthest_dist, furthest_dist, 128, 0]
		x_values = np.arange(-furthest_dist, furthest_dist+step, step)

		im = plt.imshow(plot_data, cmap=cmap,
		        aspect='auto', vmin=-vmax, vmax=vmax, 
		           interpolation='none', extent=extent)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Normalized entropy", rotation=270, va='bottom')

		_plot_peaks(plot_data, x_values, flipped=True)

		plt.xlabel(xlabel)
		ax.set_xlim(-25000, 25000)

	if title is None:
		title = f"Nucleosome entropy at firing,\nseparated origins, n={num_origins}"
	plt.title(title, 
		fontweight='demi', fontsize=15, pad=13)

