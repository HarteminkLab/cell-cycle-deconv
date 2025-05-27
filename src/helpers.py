
import numpy as np
import scipy.ndimage
from math import comb
from scipy.stats import norm
from scipy.signal import windows
import pandas as pd
import numpy as np

# The initial population mass, used in the Qr and Mgr calculations
START = 1000

# Maximum number of cell cycle "runs"
MAX_RUNS = 10

def calcH_config(config):
	return calcH(config.intervals_wt1, config.WT1_TIMEPOINTS)

def calcH(model_intervals, timepoints):
	parameters, relations, initial_timepoints, top_timepoints, bottom_timepoints, _ = model_intervals

	if len(parameters) == 8:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, halted = parameters
	else:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted = parameters

	initial_partial_H = [np.zeros((len(timepoints), len(lst)-1)) for lst in initial_timepoints]
	top_partial_H = [np.zeros((len(timepoints), len(lst)-1)) for lst in top_timepoints]
	bottom_partial_H = [np.zeros((len(timepoints), len(lst)-1)) for lst in bottom_timepoints]

	# For each timepoint in the experiment, (rows in g)
	for i, t in enumerate(timepoints):

		Q = 0

		# Compute the Qr value or mass at a given timepoint in the experiment
		# We are doing this for each run (cell cycle)
		for r in range(MAX_RUNS + 1):
			Q += Qr(mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha)

		# We also want to have a fraction of the initial population, so t=0
		# over the expected mass at our current time
		frac_init = Qr(mu0, sigma0, sigmav, delta, lambda_val, t, 0, alpha) / Q

		# Now we will construct our initial branch's columns
		# Enumerate through the timepoints of the initial branch
		for idx, tp in enumerate(initial_timepoints):

			# Compute the cdf for the initial branch timepoint
			cdf = norm.cdf(tp, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
			initial_partial_H[idx][i, :] = np.diff(cdf) * frac_init

		# For the top timepoint, we will be computing the cdf
		# to compute the mass for each timepoint interval
		# i.e.   CG1, and postG1
		# this is for the first cohort and cell cycle {0, 0}
		for runs in range(1, MAX_RUNS + 1):

			# Enumerate through the timepoints for each subinterval belonging to the to top timepoints
			for idx, tp in enumerate(top_timepoints):

				# Compute the cdf for the top branch timepoint
				cdf = norm.cdf(tp + runs * lambda_val, loc=t-mu0, scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
				top_partial_H[idx][i, :] += np.diff(cdf) * frac_init

		# This computes the cohorts for the other cohorts {1+, 1+}
		# For the top and bottom branches
		for r in range(1, MAX_RUNS + 1):
			for g in range(1, r + 1):

				# How much mass is there for the current cohort
				frac_rest = Mgr(mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha) / Q

				if frac_rest > 1e-10:
					trun_cdf = norm.cdf(r * lambda_val + (g-1) * delta - alpha, loc=t-mu0, 
						scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
					trun_denom = 1 - trun_cdf

					for idx, tp in enumerate(top_timepoints):
						for runs in range(r + 1, MAX_RUNS + 1):
							cdf = norm.cdf(tp + runs * lambda_val + g * delta, loc=t-mu0, 
									scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
							portion = cdf * 0 if trun_denom == 0 else (cdf - trun_cdf) / trun_denom
							top_partial_H[idx][i, :] += np.diff(portion) * frac_rest

					for idx, tp in enumerate(bottom_timepoints):
						cdf = norm.cdf(tp + r * lambda_val + g * delta, loc=t-mu0, 
							scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
						portion = cdf * 0 if trun_denom == 0 else (cdf - trun_cdf) / trun_denom
						bottom_partial_H[idx][i, :] += np.diff(portion) * frac_rest

	Hsegments = {}
	for i in range(len(relations)):
		relation = relations[i]

		for idx in range(1, len(relation) - 1, 2):
			label = relation[idx]
			num = int(relation[idx + 1])
			if label == 'i':
				matrix = initial_partial_H[num]
			elif label == 't':
				matrix = top_partial_H[num]
			elif label == 'b':
				matrix = bottom_partial_H[num]



			if idx == 1:
				Hsegments[i] = matrix
			else:
				Hsegments[i] += matrix

	H, Hpos, cur_start = np.hstack(list(Hsegments.values())), {}, 0
	for i in range(len(Hsegments)):
		cur_len = Hsegments[i].shape[1]
		cur_end = cur_start + cur_len
		Hpos[i] = [cur_start, cur_end]
		cur_start = cur_end

	# Scale the final matrix such that each row has an equal sum
	for i in range(H.shape[0]):
		w = np.sum(H[i, :])
		H[i, :] = H[i, :] / w


	# compute the expected alive and halted mass at each timepoint
	mass_dic = get_alive_halted_mass(model_intervals, timepoints)
	H_w_halted = np.zeros((H.shape[0], H.shape[1]+1))

	for i in range(len(timepoints)):
		time = timepoints[i]
		halted, alive, total = mass_dic[time]

		# Adjust the H matrix for the alive cells
		# columns up to the last column
		H_w_halted[i, :-1] = H[i, :]*(alive/total)

		# Add the halted cells proportion as the last column
		H_w_halted[i, -1] = halted/total

	return H_w_halted, Hpos
	

def Qr(mu0, sigma0, sigmav, delta, lambda_val, t, r, alpha):
	"""
	I believe this returns the mass of cells at a given time and reproductive instance. 
	Seemingly starting with a mass
	of 1000
	"""
	if r == 0:
		return START
	else:

		# For each of the reproductive instances r, compute the amount of mass that will contribute
		N = 0
		for i in range(r):
			normval = 1 - norm.cdf(r * lambda_val + i * delta - alpha, loc=t-mu0, 
				scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
			N += normval * START * comb(r-1, i)

		return N


def Mgr(mu0, sigma0, sigmav, delta, lambda_val, t, g, r, alpha):
	if g == 0:
		if r == 0:
			return START
		else:
			return 0
	elif g > 0:
		if r < g:
			return 0
		else:
			normval = 1 - norm.cdf(r * lambda_val + (g-1) * delta - alpha, loc=t-mu0, 
				scale=np.sqrt(sigma0**2 + t**2 * sigmav**2))
			return normval * START * comb(r-1, g-1)
	else:
		raise ValueError('g should be greater than or equal to 0.')
	
def createF(Hpos, phaseMap):
	f_partial, f_partial_list = [], {}
	for i, phase in enumerate(phaseMap):
		name = phase[0]
		subinterval_range = Hpos[phase[1]]
		indices = [e for e in range(subinterval_range[0], subinterval_range[1])]
		f_partial.extend(indices)
		f_partial_list[i] = indices
	return np.array(f_partial), f_partial_list


def get_wavelet_kernel(N, type="Symmlet", par=5):
	wavelet_kernel = WavMat(MakeONFilter(type, par), N)
	return wavelet_kernel


def WavMat(h, N, k0=None, shift=2):
	# WavMat -- Transformation Matrix of FWT_PO
	# Usage: W = WavMat(h, N, k0, shift)

	if k0 is None:
		k0 = int(np.log2(N))

	# Make QM filter G
	g = np.flip(h * (-1) ** np.arange(1, len(h) + 1))

	if not np.log2(N).is_integer():
		raise ValueError("N has to be a power of 2.")

	h = np.concatenate((h, np.zeros(N)))
	g = np.concatenate((g, np.zeros(N)))

	oldmat = np.eye(2**(int(np.log2(N)) - k0))

	for k in range(k0, 0, -1):
		ubJk = 2**(int(np.log2(N)) - k)
		ubJk1 = 2**(int(np.log2(N)) - k + 1)
		hmat = np.zeros((ubJk1, ubJk))
		gmat = np.zeros((ubJk1, ubJk))

		for jj in range(1, ubJk + 1):
			for ii in range(1, ubJk1 + 1):
				modulus = (N + ii - 2 * jj + shift) % ubJk1
				modulus = modulus + (modulus == 0) * ubJk1
				hmat[ii-1, jj-1] = h[modulus - 1]
				gmat[ii-1, jj-1] = g[modulus - 1]

		W = np.concatenate((np.matmul(oldmat, hmat.T), gmat.T))
		oldmat = W

	return W

# Example usage:
# dat = np.array([1, 0, -3, 2, 1, 0, 1, 2])
# h = MakeONFilter('Haar', 99)
# W = WavMat(h, 2**3, 3, 2)
# wt = np.dot(W, dat)
# data = np.dot(W.T, wt)
# print(wt)
# print(data)

def get_pywt_filter(wavelet_name):
	"""
	Get normalized filter coefficients from PyWavelets.
	"""
	import pywt
	wavelet = pywt.Wavelet(wavelet_name)
	# Get decomposition low-pass filter
	filter_coeffs = wavelet.dec_lo
	# Normalize
	filter_coeffs = filter_coeffs / np.linalg.norm(filter_coeffs)
	return filter_coeffs

def MakeONFilter(Type, Par):
	# ... (previous code)

	if Type == "Haar":

		f = np.array([1, 1]) / np.sqrt(2);


	elif Type == 'Symmlet':
		if Par == 4:
			f = np.array([-0.107148901418, -0.041910965125, 0.703739068656,
						  1.136658243408, 0.421234534204, -0.140317624179,
						  -0.017824701442, 0.045570345896])
		elif Par == 5:
			f = np.array([0.038654795955, 0.041746864422, -0.055344186117,
						  0.281990696854, 1.023052966894, 0.896581648380,
						  0.023478923136, -0.247951362613, -0.029842499869,
						  0.027632152958])
		elif Par == 6:
			f = np.array([0.021784700327, 0.004936612372, -0.166863215412,
						  -0.068323121587, 0.694457972958, 1.113892783926,
						  0.477904371333, -0.102724969862, -0.029783751299,
						  0.063250562660, 0.002499922093, -0.011031867509])
		elif Par == 7:
			f = np.array([0.003792658534, -0.001481225915, -0.017870431651,
						  0.043155452582, 0.096014767936, -0.070078291222,
						  0.024665659489, 0.758162601964, 1.085782709814,
						  0.408183939725, -0.198056706807, -0.152463871896,
						  0.005671342686, 0.014521394762])
		elif Par == 8:
			f = np.array([0.002672793393, -0.000428394300, -0.021145686528,
						  0.005386388754, 0.069490465911, -0.038493521263,
						  -0.073462508761, 0.515398670374, 1.099106630537,
						  0.680745347190, -0.086653615406, -0.202648655286,
						  0.010758611751, 0.044823623042, -0.000766690896,
						  -0.004783458512])
		elif Par == 9:
			f = np.array([0.001512487309, -0.000669141509, -0.014515578553,
						  0.012528896242, 0.087791251554, -0.025786445930,
						  -0.270893783503, 0.049882830959, 0.873048407349,
						  1.015259790832, 0.337658923602, -0.077172161097,
						  0.000825140929, 0.042744433602, -0.016303351226,
						  -0.018769396836, 0.000876502539, 0.001981193736])
		elif Par == 10:
			f = np.array([0.001089170447, 0.000135245020, -0.012220642630,
						  -0.002072363923, 0.064950924579, 0.016418869426,
						  -0.225558972234, -0.100240215031, 0.667071338154,
						  1.088251530500, 0.542813011213, -0.050256540092,
						  -0.045240772218, 0.070703567550, 0.008152816799,
						  -0.028786231926, -0.001137535314, 0.006495728375,
						  0.000080661204, -0.000649589896])
		else:
			raise ValueError(f"Invalid par: = {Par}")

		f = f / np.linalg.norm(f)

	return f


def get_alive_halted_mass(model_intervals, timepoints):
	
	parameters, relations, initial_timepoints, \
	top_timepoints, bottom_timepoints, _ = model_intervals
	if len(parameters) == 8:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, halted = parameters
	else:
		mu0, lambda_val, delta, sigma0, sigmav, alpha, beta, gamma1, gamma2, halted = parameters

	haltedMass = None
	mass_dic = {}
	for time in timepoints:

		Q = 0
		for r in range(MAX_RUNS + 1):
				Q += Qr(mu0, sigma0, sigmav, delta, lambda_val, time, r, alpha)
		if time == 0:
			haltedMass = Q*halted
		aliveMass = Q-haltedMass

		# dictionary of halted, alive, and total mass
		mass_dic[time] = (haltedMass, aliveMass, Q)

	return mass_dic



def compute_closest_pow2(n):
	"""Compute the closest power of 2"""
	intval = np.ceil(np.log2(n))    
	closest_pow2 = 2**intval
	return int(closest_pow2)


def pad_with_subset(f_subset):

	n = len(f_subset)
	closest_pow2 = compute_closest_pow2(n)

	# Before padding, let's see if we can mirror a subset of the start and of the indices
	# padding can get a little complicated
	missing_count = closest_pow2 - n

	missing_front = missing_count//2
	missing_end = missing_count - missing_front
	f_front_padding = f_subset[:missing_front]
	f_end_padding = f_subset[-missing_end:]
	
	f_front_padding = np.flip(f_front_padding)
	f_end_padding = np.flip(f_end_padding)
	
	f_subset_padded = np.concatenate([f_front_padding, f_subset, f_end_padding])
	return f_subset_padded


def calc_entropy(vector, base=2):
	from scipy.stats import entropy
	# compute estimated probability distribution
	total = np.sum(vector)
	prob_est = 1.* vector / (total+0.01) # avoid divide by 0
	return entropy(prob_est, base=base)


def indices_of_mapping_array(A, B):
	"""Return the indices that map the elements of B into A, where 
	A contains unique integers.
	
	This is useful when handling the H position indices. And we are operating
	on subsets of H. For example for the top branch we care about a subset of
	H. And in plotting, we want positions in B to map to the subsetted indexed array
	for only the top branch index.
	
	e.g. A = [10, 20, 30]
		 B = [20, 10]
		 
		 returns [1, 0]
	"""
	indices = [np.where(A == b)[0][0] for b in B]
	return indices


def weighted_mean(x_values, y_values):
	"""
	Calculate the weighted mean of x_values weighted by y_values.
	
	Parameters:
	x_values (np.array): array of x-axis values.
	y_values (np.array): array of weights (counts) corresponding to x_values.
	
	Returns:
	float: the weighted mean of the x_values.
	"""
	return np.average(x_values, weights=y_values)


def weighted_peak_estimation(x_values, y_values, width):
	"""
	Calculate the weighted mean of x_values within a given window around the peak (maximum y_value).
	
	Parameters:
	x_values (np.array): array of x-axis values.
	y_values (np.array): array of weights (counts) corresponding to x_values.
	width (int): window size around the peak to calculate the weighted mean.
	
	Returns:
	float: the weighted mean of the x_values within the window around the peak.
	"""
	# Identify the peak (position of the maximum y_value)
	peak_idx = np.argmax(y_values)
	peak_x_value = x_values[peak_idx]
	
	# Define the window range around the peak
	window_min = peak_x_value - width / 2
	window_max = peak_x_value + width / 2
	
	# Select values within the window
	window_mask = (x_values >= window_min) & (x_values <= window_max)
	x_window = x_values[window_mask]
	y_window = y_values[window_mask]

	# Calculate the weighted mean within the window
	if len(x_window) > 0:
		return np.average(x_window, weights=y_window)
	else:
		return peak_x_value  # If no values fall in the window, return the peak position


def common_index(arr_of_dfs, index_of_ordering):
	"""Return the common index using set logic, use df in the index of ordering
	to keep the ordering in the returned array"""
	
	order_arr_df = arr_of_dfs[index_of_ordering].copy()
	order_arr_df['index_for_ordering'] = np.arange(len(order_arr_df))
	
	intersect_set = set(order_arr_df.index)
	for df in arr_of_dfs:
		intersect_set = intersect_set.intersection(set(df.index.values))
	intersect_list = np.array(list(intersect_set))
	
	arr_df = order_arr_df.loc[intersect_list].sort_values('index_for_ordering')
	
	return arr_df.index.values
	

def get_quantile_values(dat, q):
	"""Get the quantile values and segment the input data."""
	qvals = np.quantile(dat, q=q)
	
	lower_val = float('-inf')
	
	segments = []
	lens = []
	for i in range(len(qvals)):
		
		qval = qvals[i]
		cur_seg = dat[(dat >= lower_val) & (dat < qval)]
		segments.append(cur_seg)
		
		# Update lower range
		lower_val = qval
		lens.append(len(cur_seg))
		
	# Get last segment, > qval
	cur_seg = dat[(dat >= qval)]
	segments.append(cur_seg)
	lens.append(len(cur_seg))
		
	return tuple(segments), tuple(qvals), tuple(lens)


def get_mean_between_indices(df, start, end):
	"""
	Compute the mean values between the start and end indices 
	for each row in the DataFrame.

	Parameters:
	df (pd.DataFrame): The input DataFrame.
	start (np.ndarray): The start indices.
	end (np.ndarray): The end indices.

	Returns:
	np.ndarray: A vector of mean values.
	"""
	mean_values = []

	for i in range(len(df)):
		row = df.iloc[i, start[i]:end[i]+1]
		mean_values.append(row.mean())

	return np.array(mean_values)


def combine_with_bins(data, bins, axis=1):
	"""
	Combine rows or columns of a DataFrame or 2D numpy array according to bin edges and return a new DataFrame or numpy array.

	Parameters:
	data (pd.DataFrame or np.ndarray): The input DataFrame or 2D numpy array.
	bins (array-like): The bin edges used to combine rows or columns.
	axis (int): The axis along which to combine (0 for rows, 1 for columns).

	Returns:
	pd.DataFrame or np.ndarray: A new DataFrame or numpy array with the combined rows or columns.
	"""
	# Ensure bins are sorted and unique
	import pandas as pd

	bins = np.unique(bins)
	
	# Check if the input is a DataFrame or numpy array
	if isinstance(data, pd.DataFrame):
		data_type = 'DataFrame'
	elif isinstance(data, np.ndarray):
		data_type = 'ndarray'
		data = pd.DataFrame(data)
	else:
		raise ValueError("Input data must be a pandas DataFrame or a 2D numpy array.")
	
	# Initialize an empty dictionary to store combined data
	combined_data = {}

	# Iterate over the bins to combine rows or columns
	for i in range(len(bins) - 1):
		start_idx = bins[i]
		end_idx = bins[i + 1]
		
		if axis == 1:
			# Combine columns
			combined_data[f'{start_idx}'] = data.iloc[:, start_idx:end_idx].mean(axis=1)
		elif axis == 0:
			# Combine rows
			combined_data[f'{start_idx}'] = data.iloc[start_idx:end_idx, :].mean(axis=0)
		else:
			raise ValueError("Axis must be 0 (rows) or 1 (columns).")

	# Convert the combined data to the appropriate format
	combined_df = pd.DataFrame(combined_data)
	if axis == 0:
		combined_df = combined_df.T

	if data_type == 'ndarray':
		return combined_df.to_numpy()
	return combined_df


def select_columns_by_indices(df, start_indices, end_indices):
	"""
	Select columns from the DataFrame based on the provided start and end indices for each row.

	Parameters:
	df (pd.DataFrame): The input DataFrame with columns labeled from 1 to 10 and rows indexed from 1 to 10.
	start_indices (list of int): List of start indices mapping to column names.
	end_indices (list of int): List of end indices mapping to column names.

	Returns:
	pd.DataFrame: A new DataFrame with the selected columns of shape (10x5).
	"""

	import pandas as pd

	new_data = []
	
	for row in range(len(df)):
		start = start_indices[row]
		end = end_indices[row]
		
		# if start is less than zero, add padding of nan values
		# by default if the end extends past the df's length, nas are added
		# but the same isn't true for negative values
		if start < 0:
			set_nan = True
			num_nans = -start
			start = 0
		else:
			set_nan = False
			
		selected_columns = df.iloc[row, start:end]  # Adjust for 0-based indexing
		selected_columns = selected_columns.values
		
		if set_nan:
			nans = np.repeat(np.nan, num_nans)
			selected_columns = np.concatenate([nans, selected_columns])

		new_data.append(selected_columns)
	
	# Create a new DataFrame with the selected columns
	new_df = pd.DataFrame(new_data, index=df.index)
	
	return new_df


def summarize_columns_by_indices(array, start_indices, end_indices, combine_func):
	"""
	Summarize columns from the numpy array based on the provided start and end indices for each row
	and a combining function.

	Parameters:
	array (np.ndarray): The input numpy array.
	start_indices (list of int): List of start indices for each row.
	end_indices (list of int): List of end indices for each row.
	combine_func (callable): Function to combine the selected columns (e.g., np.mean, np.sum).

	Returns:
	np.ndarray: A 1D array with the summarized values for each row.
	"""
	
	summarized_values = np.array([
		combine_func(array[row, start:end]) 
		for row, (start, end) in enumerate(zip(start_indices, end_indices))
	])
	
	return summarized_values


def get_equal_partitions(vec, k):
	"""Compute equal partitions of a given vector. Returns 
	a vector of the start and indices of each partition"""
	n = len(vec)
	bin_edges = np.arange(0, n, n/k)
	bin_edges = np.concatenate([bin_edges.astype(int), np.array([n-1])])
	partition_indices = []
	for i in range(1, len(bin_edges)):
		partition_indices.append((bin_edges[i-1], bin_edges[i]))
	return partition_indices


def normalize_sum_ndarray(input_arr, axis=1):

	if type(input_arr) == pd.DataFrame:
		df = input_arr
		arr = df.values

	if axis == 1:
		arr = arr / arr.mean(axis=axis).reshape((-1, 1))
	else:
		arr = arr / arr.mean(axis=axis).reshape((1, -1))

	if type(input_arr) == pd.DataFrame:
		return_df = pd.DataFrame(arr, index=df.index)
		return_df.columns = df.columns
		return return_df

	return arr


def normalize_max_min(dat, indices=None):
	"""Normalize the input data to the min and max for comparing"""

	if indices is None: indices = np.arange(len(dat))

	min_v, max_v = dat[indices].min(), dat[indices].max()
	delta = (max_v - min_v) + 1e-5 # avoid divide by zero
	dat = dat.copy()
	dat = (dat - min_v) / delta
	return dat


def create_gaussian_kernel(size, sigma):
	"""Create a 2d kernel for smoothing"""
	x = np.linspace(- (size // 2), size // 2, size)
	y = np.linspace(- (size // 2), size // 2, size)
	x, y = np.meshgrid(x, y)
	kernel = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
	kernel /= np.sum(kernel)
	return kernel


def smooth_matrix(matrix, kernel):
	"""Smooth an input 2d matrix with a 2d kernel"""
	smoothed_matrix = scipy.ndimage.convolve(matrix, kernel, mode='reflect')
	return smoothed_matrix


def smooth_data(img, size=5, sigma=0.75):

	# Create a 2D Gaussian kernel
	gaussian_kernel = create_gaussian_kernel(size, sigma)

	# Apply Gaussian smoothing to the matrix
	img = smooth_matrix(img, gaussian_kernel)

	return img


def proportion_indices(indices, props):
	return np.array([indices[int(prop * len(indices))] for prop in props])


def smooth_transitions_custom(data, window_size=3, sigma=0.5, power=1, axis=-1):
	"""Apply smoothing along specified axis while preserving min/max values."""
	window = windows.general_gaussian(window_size, power, sigma)
	window = window / window.sum()
	
	def smooth_1d(x):

		if x.sum() == 0: return x

		padded = np.pad(x, (window_size//2, window_size//2), mode='edge')
		smoothed = np.convolve(padded, window, mode='valid')

		if (smoothed.max() - smoothed.min()) == 0: return x

		# Scale to original range
		return (smoothed - smoothed.min()) * (x.max() - x.min()) / (smoothed.max() - smoothed.min()) + x.min()
	
	return np.apply_along_axis(smooth_1d, axis, data)


def midpoints(arr):
	return (arr[:-1] + arr[1:]) / 2


def get_level_based_weights(N, scale=2):
	"""Weight the coefficients by the level of of the wavelet. This
	discourages high-frequency coefficients. Exponentially increasing
	weighting based on coefficient level."""
	
	weights = np.ones(N)
	# levels = int(np.log2(N))

	# weights[N//2:] = 1

	# for level in range(1, levels):
	# 	start_index = 2**(level)
	# 	end_index= 2**(level+1)
	# 	weight = scale**(level+1)
	# 	weights[start_index:end_index] = weight

	return weights


def compute_branch_lengths(config1):
	# Compute the length of each of the smoothing terms

	rg1_tps = config1.get_timepoints_for_phase('RG1')
	cg1_tps = config1.get_timepoints_for_phase('CG1')
	dg1_tps = config1.get_timepoints_for_phase('DG1')
	postg1_tps = config1.get_timepoints_for_phase('postG1')

	length_rg1 = rg1_tps[-1]-rg1_tps[0]
	length_cg1 = cg1_tps[-1]-cg1_tps[0]
	length_dg1 = dg1_tps[-1]-dg1_tps[0]
	length_postg1 = postg1_tps[-1]-postg1_tps[0]

	length_rg1, length_cg1, length_dg1, length_postg1

	recovery_smoothing_tps_length = length_rg1+length_postg1
	top_smoothing_tps_length = length_cg1+length_postg1+length_postg1
	bottom_smoothing_tps_length = length_dg1+length_postg1+length_postg1

	print("Length of of the padded branches:", 
		  recovery_smoothing_tps_length,
		  top_smoothing_tps_length, 
		  bottom_smoothing_tps_length)

	print("1/Proportion of the daughter branch (longest):", 
		  bottom_smoothing_tps_length/recovery_smoothing_tps_length,
		  bottom_smoothing_tps_length/top_smoothing_tps_length, 
		  bottom_smoothing_tps_length/bottom_smoothing_tps_length)


def downsample_bins(bin_data, bin_size=(10, 10)):
	"""
	Parameters:
	bin_data (numpy.ndarray): Input 3D array with shape (time_points, height, width)
	bin_size (tuple): Size of each bin as (y_bin_size, x_bin_size), default is (10, 10)
	
	Returns:
	numpy.ndarray: Downsampled array with preserved mean values
	"""

	num_timepoints, y_dim, x_dim = bin_data.shape
	y_bin_size, x_bin_size = bin_size
	
	# Calculate padding needed
	y_padding = (y_bin_size - (y_dim % y_bin_size)) % y_bin_size
	x_padding = (x_bin_size - (x_dim % x_bin_size)) % x_bin_size
	
	# Target dimensions after padding
	padded_y_dim = y_dim + y_padding
	padded_x_dim = x_dim + x_padding
	
	# Calculate output dimensions
	target_y_dim = padded_y_dim // y_bin_size
	target_x_dim = padded_x_dim // x_bin_size
	
	# Create padded array with mean padding if needed
	if y_padding > 0 or x_padding > 0:
		pad_value = 0
		padded_data = np.full((num_timepoints, padded_y_dim, padded_x_dim), pad_value, dtype=bin_data.dtype)
		padded_data[:, :y_dim, :x_dim] = bin_data
	else:
		padded_data = bin_data
	
	# Reshape and compute mean in one go
	# First reshape to separate bins: (time, target_y, bin_y, target_x, bin_x)
	reshaped = padded_data.reshape(
		num_timepoints,
		target_y_dim, y_bin_size,
		target_x_dim, x_bin_size
	)
	
	# Then take mean over bin dimensions
	downsampled = reshaped.mean(axis=(2, 4))
	
	return downsampled


import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def compute_row_correlations(df1, df2):
	"""
	Compute Pearson correlation coefficients for each corresponding row in two dataframes.
	
	Parameters:
	-----------
	df1, df2 : pandas.DataFrame
		DataFrames with the same shape and index
		
	Returns:
	--------
	pandas.DataFrame with correlation coefficients and p-values
	"""
	from scipy.stats import pearsonr

	# Check that dataframes have the same shape
	if df1.shape != df2.shape:
		raise ValueError("DataFrames must have the same shape")
	
	# Initialize lists to store results
	correlations = []
	p_values = []
	indices = []
	
	# Compute correlation for each row
	for idx in df1.index:
		corr, p_val = pearsonr(df1.loc[idx], df2.loc[idx])
		correlations.append(corr)
		p_values.append(p_val)
		indices.append(idx)
	
	# Create a results dataframe
	results = pd.DataFrame({
		'correlation': correlations,
		'p_value': p_values
	}, index=indices)
	
	return results


def plot_raw_timepoints(config, s_color='red', flip=False, lw=0.5):
	"""Plot vertical lines along helpful dilineations of the config's cell cycle
	points"""
	import matplotlib.pyplot as plt
	first_s_start, first_s_end, end_of_first_lambd, \
		cg1_length, s_length, lambda_len = \
		config.get_key_timepoints_in_raw()

	if flip: 
		plot_func = plt.axhline
	else:
		plot_func = plt.axvline

	# First cell cycle
	for time in [first_s_start, first_s_end]:
		plot_func(time, c=s_color, lw=lw, ls='solid')

	plot_func(end_of_first_lambd, c='black', lw=lw, ls='solid')
	
	# Second cell cycle
	for time in [end_of_first_lambd+cg1_length, 
		end_of_first_lambd+cg1_length+s_length]:
		plot_func(time, c=s_color, lw=lw, ls='solid')

	if end_of_first_lambd+lambda_len <= config.timepoints[-1]:
		plot_func(end_of_first_lambd+lambda_len, c='black', lw=lw, ls='solid')


def interpolate_increase_length(vec, new_size):
    """
    Interpolate a vector to increase its length using linear interpolation.

    Current usage: Computing the correlation between two runs of F with varying branch length.
    Applicable in the find optimal alpha stage when individual replicates are computing different
    alpha values with varying G1 length (number of G1 indices).
    """
    vec = np.array(vec)
    
    if new_size <= len(vec):
        raise ValueError(f"new_size ({new_size}) must be larger than current size ({len(vec)})")
    
    # Create original indices (0, 1, 2, ..., n-1)
    old_indices = np.arange(len(vec))
    
    # Create new indices (0, 0.5, 1, 1.5, ..., n-1) scaled appropriately
    new_indices = np.linspace(0, len(vec) - 1, new_size)
    
    # Interpolate
    interpolated_vec = np.interp(new_indices, old_indices, vec)
    
    return interpolated_vec