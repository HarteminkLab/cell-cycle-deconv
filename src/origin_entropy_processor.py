
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

import numpy as np
from matplotlib import pyplot as plt


def _accumulate_branch(per_origin_2d, firing_idx_window, branch_len,
					   anchor, master_h, shift0):
	"""Accumulate one branch's origins into a shared master grid.

	per_origin_2d : (n_origins, n_distances, branch_len) for this branch.
	firing_idx_window : per-origin firing index already translated into this
		branch's window coords (0..branch_len). NaN / out-of-range dropped.
	anchor, master_h, shift0 : shared grid geometry (same for both branches),
		so the two branches' composites are row-aligned and can be averaged.

	Returns accum, count (both master_h x n_distances) and n_used.
	"""
	firing_idx_window = np.asarray(firing_idx_window, dtype=float)
	valid = ~np.isnan(firing_idx_window)
	valid &= (firing_idx_window >= 0) & (firing_idx_window < branch_len)
	per_origin_2d = per_origin_2d[valid]
	firing_idx = np.round(firing_idx_window[valid]).astype(int)
	n_used = len(firing_idx)

	n_distances = per_origin_2d.shape[1]
	accum = np.zeros((master_h, n_distances))
	count = np.zeros((master_h, n_distances))

	offsets = anchor - firing_idx                  # +ve pushes early origins down
	for j in range(n_used):
		start = offsets[j] + shift0
		# Skip any origin that would fall outside the shared grid
		if start < 0 or start + branch_len > master_h:
			continue
		slice_jt = per_origin_2d[j].T              # -> (branch_len, n_distances)
		accum[start:start + branch_len] += slice_jt
		count[start:start + branch_len] += 1

	return accum, count, n_used


def _build_offset_composite_bt(b_dat, b_firing_window, t_dat, t_firing_window,
							   branch_len):
	"""Align each branch to a common anchor in its own window coords, then
	average the two aligned composites.

	Both branches share one master grid so their firing rows coincide; the
	average is therefore "rows relative to firing" for both.
	"""
	# Shared anchor and grid geometry, computed from both branches' firing
	# indices pooled, so neither branch clips.
	all_idx = np.concatenate([
		np.round(np.asarray(b_firing_window, dtype=float)),
		np.round(np.asarray(t_firing_window, dtype=float)),
	])
	all_idx = all_idx[~np.isnan(all_idx)]
	all_idx = all_idx[(all_idx >= 0) & (all_idx < branch_len)].astype(int)

	anchor = int(round(all_idx.mean()))
	max_off = anchor - all_idx.min()               # largest downward shift
	min_off = anchor - all_idx.max()               # most negative shift
	shift0 = -min_off
	master_h = branch_len + (max_off - min_off)

	b_accum, b_count, n_b = _accumulate_branch(
		b_dat, b_firing_window, branch_len, anchor, master_h, shift0)
	t_accum, t_count, n_t = _accumulate_branch(
		t_dat, t_firing_window, branch_len, anchor, master_h, shift0)

	# Merge the two branches: sum contributions and counts across b and t,
	# so each master cell is the mean over all origin-branches reaching it.
	accum = b_accum + t_accum
	count = b_count + t_count

	with np.errstate(invalid='ignore', divide='ignore'):
		composite = accum / count
	composite[count == 0] = np.nan

	# Crop ragged edges. count now pools both branches, so the natural
	# threshold is half of the per-branch origin count, summed.
	n_used = max(n_b, n_t)
	count_per_row = count[:, 0]
	threshold = max(1, n_used // 2)
	keep = count_per_row >= threshold
	first_kept = np.argmax(keep)

	composite = composite[keep]
	count_per_row = count_per_row[keep]
	anchor_row = (anchor + shift0) - first_kept

	return composite, anchor_row, count_per_row, n_b, n_t


def plot_entropy_by_distance(deconv_all_origin_dist_entropies, config1,
	folded=True, furthest_dist=20000, step=1000, title=None, xlabel="Distance from origin"):

	vmax = 1.25

	plt.figure(figsize=(4, 5))

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

	cmap = 'magma'

	if folded:
		deconv_folded_data = fold_halves_together(tb_data)
		plot_data_folded = _normed_data(deconv_folded_data.mean(0)).T

		x_values = np.arange(0, furthest_dist+step, step)
		extent = [-step_2, furthest_dist+step_2, 128, 0]

		im = plt.imshow(plot_data_folded, cmap=cmap,
				aspect='auto', 
				vmin=-vmax, vmax=vmax, 
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
		# ax.set_xlim(-25000, 25000)

	if title is None:
		title = f"Nucleosome entropy at firing,\nseparated origins, n={num_origins}"
	plt.title(title, 
		fontweight='demi', fontsize=15, pad=13)


import numpy as np
from matplotlib import pyplot as plt

def _accumulate_branch(per_origin_2d, firing_idx_window, branch_len,
					   anchor, master_h, shift0):
	"""Accumulate one branch's origins into a shared master grid.

	per_origin_2d : (n_origins, n_distances, branch_len) for this branch.
	firing_idx_window : per-origin firing index already translated into this
		branch's window coords (0..branch_len). NaN / out-of-range dropped.
	anchor, master_h, shift0 : shared grid geometry (same for both branches),
		so the two branches' composites are row-aligned and can be averaged.

	Returns accum, count (both master_h x n_distances) and n_used.
	"""
	firing_idx_window = np.asarray(firing_idx_window, dtype=float)
	valid = ~np.isnan(firing_idx_window)
	valid &= (firing_idx_window >= 0) & (firing_idx_window < branch_len)
	per_origin_2d = per_origin_2d[valid]
	firing_idx = np.round(firing_idx_window[valid]).astype(int)
	n_used = len(firing_idx)

	n_distances = per_origin_2d.shape[1]
	accum = np.zeros((master_h, n_distances))
	count = np.zeros((master_h, n_distances))

	offsets = anchor - firing_idx                  # +ve pushes early origins down
	for j in range(n_used):
		start = offsets[j] + shift0
		# Skip any origin that would fall outside the shared grid
		if start < 0 or start + branch_len > master_h:
			continue
		slice_jt = per_origin_2d[j].T              # -> (branch_len, n_distances)
		accum[start:start + branch_len] += slice_jt
		count[start:start + branch_len] += 1

	return accum, count, n_used


def _build_offset_composite_bt(b_dat, b_firing_window, t_dat, t_firing_window,
							   branch_len):
	"""Align each branch to a common anchor in its own window coords, then
	average the two aligned composites.

	Both branches share one master grid so their firing rows coincide; the
	average is therefore "rows relative to firing" for both.
	"""
	# Shared anchor and grid geometry, computed from both branches' firing
	# indices pooled, so neither branch clips.
	all_idx = np.concatenate([
		np.round(np.asarray(b_firing_window, dtype=float)),
		np.round(np.asarray(t_firing_window, dtype=float)),
	])
	all_idx = all_idx[~np.isnan(all_idx)]
	all_idx = all_idx[(all_idx >= 0) & (all_idx < branch_len)].astype(int)

	anchor = int(round(all_idx.mean()))
	max_off = anchor - all_idx.min()               # largest downward shift
	min_off = anchor - all_idx.max()               # most negative shift
	shift0 = -min_off
	master_h = branch_len + (max_off - min_off)

	b_accum, b_count, n_b = _accumulate_branch(
		b_dat, b_firing_window, branch_len, anchor, master_h, shift0)
	t_accum, t_count, n_t = _accumulate_branch(
		t_dat, t_firing_window, branch_len, anchor, master_h, shift0)

	# Merge the two branches: sum contributions and counts across b and t,
	# so each master cell is the mean over all origin-branches reaching it.
	accum = b_accum + t_accum
	count = b_count + t_count

	with np.errstate(invalid='ignore', divide='ignore'):
		composite = accum / count
	composite[count == 0] = np.nan

	# Crop ragged edges. count now pools both branches, so the natural
	# threshold is half of the per-branch origin count, summed.
	n_used = max(n_b, n_t)
	count_per_row = count[:, 0]
	threshold = max(1, n_used // 2)
	keep = count_per_row >= threshold
	first_kept = np.argmax(keep)

	composite = composite[keep]
	count_per_row = count_per_row[keep]
	anchor_row = (anchor + shift0) - first_kept

	return composite, anchor_row, count_per_row, n_b, n_t


def plot_entropy_by_distance_offset(deconv_all_origin_dist_entropies, config1,
	origin_timings, folded=True, furthest_dist=20000, step=1000,
	title=None, xlabel="Distance from origin"):
	"""Same heatmap as plot_entropy_by_distance, but each origin is offset
	along the time axis by its replication firing index (mean of the b and t
	branch indices) before averaging, so firing aligns across origins.

	deconv_all_origin_dist_entropies is the (n_origins, n_distances,
	num_timepoints) matrix from generate_origin_entropy_matrix; origin_timings
	must be row-aligned to it and carry replication_index_b / replication_index_t.
	"""

	vmax = 1.25

	plt.figure(figsize=(4, 5))

	# --- b/t branch slices (kept separate; aligned per branch) ---
	b_sel = np.asarray(config1.b_indices())
	t_sel = np.asarray(config1.t_indices())
	b_indices_dat = deconv_all_origin_dist_entropies[:, :, b_sel]
	t_indices_dat = deconv_all_origin_dist_entropies[:, :, t_sel]
	branch_len = b_indices_dat.shape[2]            # 128

	# --- firing index per branch, translated into that branch's window ---
	# Indices live on the full 0..256 axis; each branch is a contiguous
	# window starting at its first selected timepoint, so subtract that start.
	b_start = b_sel.min()
	t_start = t_sel.min()
	b_firing = origin_timings['replication_index_b'].values - b_start
	t_firing = origin_timings['replication_index_t'].values - t_start

	# --- align each branch to a shared anchor, then average ---
	composite, anchor_row, count_per_row, n_b, n_t = _build_offset_composite_bt(
		b_indices_dat, b_firing, t_indices_dat, t_firing, branch_len)
	n_used = max(n_b, n_t)

	cmap = 'magma'
	step_2 = step // 2

	# x ticks identical to original
	tick_step = 10000
	tick_start = (furthest_dist // 5000) * 5000
	xticks = np.arange(-tick_start, tick_start + tick_step, tick_step)
	xtick_labels = [f"{x/step:.0f} kb" if x <= 0 else
		f"+{x/step:.0f} kb" for x in xticks]
	plt.xticks(xticks, xtick_labels)

	if folded:
		# fold on the distance axis exactly as before — composite is now
		# (rows, distances), so axis=1 is still distance
		folded_composite = fold_halves_together(composite, axis=1)
		plot_data_folded = _normed_data(folded_composite.T).T

		x_values = np.arange(0, furthest_dist + step, step)
		extent = [-step_2, furthest_dist + step_2, composite.shape[0], 0]

		plt.imshow(plot_data_folded, cmap=cmap, aspect='auto',
			vmin=-vmax, vmax=vmax,
			 extent=extent, interpolation='none')
		cbar = plt.colorbar()

		_plot_peaks(plot_data_folded, x_values, flipped=True)
		plt.xlim(0, 40000)
		cbar.ax.set_ylabel("Normalized entropy", rotation=270, va='bottom')
		plt.xlabel(xlabel)

	else:
		plot_data = _normed_data(composite.T).T

		extent = [-furthest_dist, furthest_dist, composite.shape[0], 0]
		x_values = np.arange(-furthest_dist, furthest_dist + step, step)

		plt.imshow(plot_data, cmap=cmap, aspect='auto',
			vmin=-vmax, vmax=vmax, interpolation='none', extent=extent)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Normalized entropy", rotation=270, va='bottom')

		_plot_peaks(plot_data, x_values, flipped=True)
		plt.xlabel(xlabel)
		#plt.gca().set_xlim(-25000, 25000)

	# Mark the firing anchor; drop the cell-cycle phase labels for now
	ax = plt.gca()
	ax.axhline(anchor_row, color='black', lw=0.8, ls='--', alpha=0.6)
	ax.set_yticks([anchor_row])
	ax.set_yticklabels(['firing'], fontsize=8)
	ax.set_ylabel("Time relative to firing")

	if title is None:
		title = (f"Nucleosome entropy aligned\nto replication time,\n"
				 f"efficient origins, n={len(origin_timings)}")
	plt.title(title, fontweight='demi', fontsize=15, pad=13)