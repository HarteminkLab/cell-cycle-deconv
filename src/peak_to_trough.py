
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def combine_ptr_score(c, d, weight):
	score = np.power(c, weight)*np.power(d, 1-weight);
	return score


def compute_quantile_ptr_2d(arr2d, axis=1):
	ptr_mat = np.apply_along_axis(compute_quantile_ptr, axis, arr2d)
	return ptr_mat


def compute_quantile_ptr(data_f, lo=0.2, hi=0.8, eps=1, return_indices=False):
	"""
	Compute the 80/20 PTR of the data, ensuring no division by 0 by adding a small pseudo count.
	Use 'nearest' interpolation for quantile calculation and optionally return the indices of
	low and high quantile values.
	"""
	# Use np.quantile with 'nearest' interpolation to get the lo and hi values directly
	f_lo = np.quantile(data_f, lo, interpolation='nearest')
	f_hi = np.quantile(data_f, hi, interpolation='nearest')

	# Find the indices of these nearest values in the original data
	lo_val_idx = np.where(data_f == f_lo)[0][0]  # Taking the first match
	hi_val_idx = np.where(data_f == f_hi)[0][0]  # Taking the first match

	# Add a pseudo count to prevent divide by zero
	f_hi_adjusted = f_hi + eps
	f_lo_adjusted = f_lo + eps
	
	# Calculate PTR
	ptr = f_hi_adjusted / f_lo_adjusted
	
	if return_indices:
		return ptr, lo_val_idx, hi_val_idx, f_lo, f_hi
	else:
		return ptr


def compute_ptr_f(config, f, quantiles=[0.2, 0.8]):
	f_ptrs = np.zeros(f.shape[1])
	for i in range(f.shape[1]):
		cptr, dpt, ptr = compute_ptr(config, f[:, i], quantiles[0], quantiles[1])
		f_ptrs[i] = ptr
	return f_ptrs


def compute_ptr_f_top(config, f, quantiles=[0.2, 0.8]):
	f_ptrs = np.zeros(f.shape[1])
	for i in range(f.shape[1]):
		cptr, dpt, ptr = compute_ptr(config, f[:, i], quantiles[0], quantiles[1])
		f_ptrs[i] = ptr
	return f_ptrs


def compute_max_min_locations(config, gene_f, ret_all=False):
	"""Once we have 80/20 ptr, we are interested in where the absolute max and min locations
	are for plotting


	Returns for max and min as a tuple:
		- The min/max value
		- The min/max index in F (or columns in H)
		- The timepoint of the min/max
		- The phase of the min/max

	e.g.

	return (min_tuple of above), (max_tuple of above)


	if ret_all is True:
		return (min_tuple of above), (max_tuple of above), (mother min and max tuple), (daughter min and max tuple)

	"""

	c_indices = config.get_Hpositions_for_branch('t')
	d_indices = config.get_Hpositions_for_branch('b')
	c_timepoints = config.get_timepoints_for_branch('t')
	d_timepoints = config.get_timepoints_for_branch('b')

	(cg1_timepoints, c_s_timepoints, c_g2m_timepoints), \
    (dg1_timepoints, d_s_timepoints, d_g2m_timepoints) = config.get_phase_timepoints_for_plotting()

	c_tps_mapping = {"C_G1": cg1_timepoints, 
					"C_S": c_s_timepoints, 
					"C_G2M": c_g2m_timepoints}

	d_tps_mapping = {"D_G1": dg1_timepoints, 
					"D_S": d_s_timepoints, 
					"D_G2M": d_g2m_timepoints}

	def get_min_max_indices(f_array, indices, timepoints, timepoints_mapping):
		"""
		Get the min and max index tp and phase for a F vector of a gene
		"""

		subset_f = f_array[indices]

		def get_min_or_max_results(subset_f, indices, func):
			"""
			Get the min or max index, tp, and phase for a gene's f array of gene expression.

			Performs the mapping from columns in F to absolute indices for lookup in the
			timepoints array for the given branch, then uses the timepoints mapping
			to get the specific phase of the min or max value.
			"""

			# Compute the argmin or argmax and mapping into 
			# F
			func_index = func(subset_f)
			f_index_func = indices[func_index]
			func_val = subset_f[func_index]

			func_tp = timepoints[func_index]

			# Which interval does the tp lie in?
			func_phase = None
			for phase, tps in timepoints_mapping.items():
				if func_tp >= tps[0] and func_tp <= tps[-1]:
					func_phase = phase

			func_res = (func_val, f_index_func, func_tp, func_phase)

			return func_res

		min_res = get_min_or_max_results(subset_f, indices, np.argmin)
		max_res = get_min_or_max_results(subset_f, indices, np.argmax)

		return (min_res, max_res)
		

	# Get the min and maxes for mother and daughter
	(c_min_res, c_max_res) = get_min_max_indices(gene_f, c_indices, c_timepoints, c_tps_mapping)
	(d_min_res, d_max_res) = get_min_max_indices(gene_f, d_indices, d_timepoints, d_tps_mapping)

	c_max_val = c_max_res[0]
	c_min_val = c_min_res[0]
	d_max_val = d_max_res[0]
	d_min_val = d_min_res[0]

	# Get the absolute max between the two
	if c_max_val > d_max_val:
		max_res = c_max_res
	else:
		max_res = d_max_res

	# Set the minimum
	if c_min_val < d_min_val:
		min_res = c_min_res
	else:
		min_res = d_min_res

	if ret_all:
		ret = (min_res, max_res), (c_min_res, c_max_res), (d_min_res, d_max_res)
	else:
		ret = (min_res, max_res)

	return ret


def compute_ptr(config, gene_f, lo=0.2, hi=0.8):

	c_indices = config.get_Hpositions_for_branch('t')
	cg1_f = gene_f[c_indices]

	c_timepoints = config.get_timepoints_for_branch('t')
	_, scaled_cg1_f, mapping_cg1 = rescale_with_mapping(c_timepoints, cg1_f)

	cptr = compute_quantile_ptr(scaled_cg1_f, lo, hi)

	return cptr


def compute_ptr_distinct_cg1_dg1(config, gene_f, lo=0.2, hi=0.8, return_indices=False):

	c_indices = config.get_Hpositions_for_branch('t')
	d_indices = config.get_Hpositions_for_branch('b')

	cg1_f = gene_f[c_indices]
	dg1_f = gene_f[d_indices]

	c_timepoints = config.get_timepoints_for_branch('t')
	d_timepoints = config.get_timepoints_for_branch('b')

	_, scaled_cg1_f, mapping_cg1 = rescale_with_mapping(c_timepoints, cg1_f)
	_, scaled_dg1_f, mapping_dg1 = rescale_with_mapping(d_timepoints, dg1_f)

	weight = 2./3.;

	# These indices are based on the input scaled f vector, not the
	# indices of the timepoints, so we'll need to convert them
	cptr, c_l_idx, c_h_idx, c_lo, c_hi = compute_quantile_ptr(scaled_cg1_f, lo, hi, return_indices=True)
	dptr, d_l_idx, d_h_idx, d_lo, d_hi = compute_quantile_ptr(scaled_dg1_f, lo, hi, return_indices=True)

	# Unscale the hi and lo mappings
	c_h_idx = mapping_cg1[c_h_idx]
	c_l_idx = mapping_cg1[c_l_idx]

	d_h_idx = mapping_dg1[d_h_idx]
	d_l_idx = mapping_dg1[d_l_idx]

	c_lo_tp, c_hi_tp, d_lo_tp, d_hi_tp = (c_timepoints[c_l_idx], c_timepoints[c_h_idx], 
		d_timepoints[d_l_idx], d_timepoints[d_h_idx])

	# convert the returned indices into the indices that match to the F indices in CG1 and DG1
	c_l_idx = c_indices[c_l_idx]
	c_h_idx = c_indices[c_h_idx]
	d_l_idx = d_indices[d_l_idx]
	d_h_idx = d_indices[d_h_idx]

	combinedPtr = combine_ptr_score(cptr, dptr, weight)

	if return_indices:
		return cptr, dptr, combinedPtr, \
			c_l_idx, c_h_idx, d_l_idx, d_h_idx, \
			c_lo, c_hi, d_lo, d_hi, \
			c_lo_tp, c_hi_tp, d_lo_tp, d_hi_tp

	return cptr, dptr, combinedPtr


def rescale_with_mapping(x, y, interval=1):
	"""
	Rescale x and y data points based on a specified interval and return a mapping from
	new indices to old indices.

	Parameters:
	x (array-like): Original x data points.
	y (array-like): Original y data points.
	interval (float, optional): Interval for rescaling. Default is 1.

	Returns:
	newx (numpy.ndarray): Rescaled x data points.
	newy (numpy.ndarray): Rescaled y data points.
	mapping (list): Mapping of new indices to old indices.
	"""
	if len(x) < 2 or len(y) < 2:
		raise ValueError("Expect at least 2 data points in both x and y")

	len_new = int(np.ceil((x[-1] - x[0]) / interval))
	newx = np.linspace(x[0], x[0] + len_new * interval, len_new + 1)
	newy = np.zeros(len_new + 1)
	mapping = np.zeros(len_new + 1, dtype=int)

	for idx in range(len_new):
		cur_x = newx[idx]
		# Find the left and the right neighbors
		left_pos = np.where(x <= cur_x)[0][-1]
		right_pos = np.where(x >= cur_x)[0][0]

		# Determine the closest original index for mapping
		if cur_x - x[left_pos] < x[right_pos] - cur_x:
			mapping[idx] = left_pos
		else:
			mapping[idx] = right_pos

		x_left = x[left_pos]
		y_left = y[left_pos]
		x_right = x[right_pos]
		y_right = y[right_pos]

		if x_left == x_right:
			newy[idx] = y_left
		else:
			newy[idx] = y_left + (cur_x - x_left) / (x_right - x_left) * (y_right - y_left)

	# Extend the last point as in the original function
	newx[-1] = newx[-2] + interval
	newy[-1] = newy[-2]
	mapping[-1] = mapping[-2]  # Map the last new index to the closest original index

	return newx, newy, mapping


def get_chrom_g_ptr(gene_g_chrom):
	"""
	Compute the chromatin ptr for g for yulong replicate 2 

	Assumes 15 timepoints and we should skip the first two for recovery in the 
	ptr calculation
	"""

	# TODO: Hard-coded (15 timepoints)
	gene_g_chrom = gene_g_chrom.reshape(15, -1)
	gene_g_ptrs = np.zeros((gene_g_chrom.shape[1], 1))

	# Compute the PTR for each grid element in the G matrix
	for i in range(gene_g_chrom.shape[1]):

		# Skip the first 2 timepoints (as recovery is 23 minutes)
		ptr = compute_80_20_ptr(gene_g_chrom[2:, i])
		gene_g_ptrs[i] = ptr

	return gene_g_ptrs, np.max(gene_g_ptrs)
