
import numpy as np
import pandas as pd

from src.timer import Timer
from matplotlib import pyplot as plt
from src.read_bam import read_mnase_bam
from src.figure_configs import FiguresConfig


def select_F_imgs_in_window(F, F_span, select_span):
	"""For a given deconvolved F matrix, with a known span.
	Select a subset of the span. Useful for selecting the gene related
	features in a window"""
	span_translated = select_span[0]-F_span[0], select_span[1]-F_span[0]
	span_bins = int(span_translated[0]//10), int(span_translated[1]//10)
	F_imgs = F.reshape((F.shape[0], 26, -1))        
	selected_F_bins = F_imgs[:, :, span_bins[0]:span_bins[1]]
	return selected_F_bins


def retrieve_fragment_length_selection_curves(plot=True):
	"""Defines fragment length curves using the length distribution. This is
	will select an approximation of the fragment lengths relevant for nucleosome and
	small fragment occupancy without the use of hard cutoffs"""

	from src.mnase_normalization import load_target_distribution
	from scipy.stats.distributions import norm

	target_len_dist = load_target_distribution(verbose=False)
	if plot:	
		plt.figure(figsize=(6, 4))
		plt.plot(target_len_dist, label="Raw MNase length distribution", lw=4,
				c='gray')

	xs = target_len_dist.index
	ys = norm.pdf(xs, loc=165, scale=17.8)
	nucleosome_curve = pd.DataFrame(ys, index=xs)

	ys = ys / ys.max() * target_len_dist.max()

	if plot:
		plt.plot(xs, ys, label="Nucleosome fragment selection", lw=2,
				c=plt.cm.Purples(0.75))

	xs = target_len_dist.index
	ys = norm.pdf(xs, loc=80, scale=19.5)
	ys = ys / ys.max() * target_len_dist[0:100].max()
	small_fragments_curve = pd.DataFrame(ys, index=xs)

	if plot:
		plt.plot(xs, ys, label="Small fragment selection", lw=2,
				c=plt.cm.Oranges(0.75))
		plt.xlabel("Fragment length")
		plt.ylabel("Frequency")
		plt.legend()
		plt.xlim(0, xs[-1])
		plt.title("Fragment length selection curves")

	xs = target_len_dist.index
	ys = norm.pdf(xs, loc=124, scale=18)
	ys = ys / ys.max() * target_len_dist[0:120].max()
	intermediate_curve = pd.DataFrame(ys, index=xs)

	# Normalize the curves	
	nucleosome_curve = nucleosome_curve / nucleosome_curve.mean()
	small_fragments_curve = small_fragments_curve / small_fragments_curve.mean()

	return nucleosome_curve, intermediate_curve, small_fragments_curve


def downsample_kernel(kernel):
	"""Reshape and downsample kernel for selection of mnase reads"""
	from src.helpers import downsample_bins
	kernel_reshaped = kernel.reshape((1, -1, 1))
	downsampled_kernel = downsample_bins(kernel_reshaped, (10, 1))[0]
	return downsampled_kernel


def select_w_kernel(imgs, kernel, normalize=True):
	"""User correlation to apply the kernel to the image to select the appropriate
	reads. Assume the input data is normalized, so normalize resulting selection"""
	from src.helpers import downsample_bins
	from scipy.signal import correlate2d
	kernel_selected = np.zeros((imgs.shape[0], imgs.shape[2]))
	for i in range(imgs.shape[0]):
		img = imgs[i]
		res = correlate2d(img, kernel, mode='valid')   
		kernel_selected[i] = res
	if normalize:
		kernel_selected = kernel_selected / kernel_selected.mean()
	return kernel_selected


def load_downsampled_selection_kernel(fragment_type):
	# Get the appropriate fragment length selection curve
	nuc_curve, intermediate_curve, small_curve = retrieve_fragment_length_selection_curves(plot=False)
	
	# Select the appropriate kernel based on fragment_type parameter
	if fragment_type.lower() == "nucleosome":
		kernel = downsample_kernel(nuc_curve.values)
	elif fragment_type.lower() == "small":
		kernel = downsample_kernel(small_curve.values)
	else:
		raise ValueError(f"Unknown fragment type: {fragment_type}. Use 'nucleosome' or 'small'.")

	return kernel


def compute_chromatin_metric(F, fragment_type="nucleosome", metric_type="entropy"):
	"""
	Compute chromatin metrics for a specified region.
	
	Args:
		F: The chromatin data matrix with dimensions (timepoints, fragment_lengths, positions)
		   This should already be the correct region of interest
		fragment_type: Type of fragments to analyze - "nucleosome" or "small"
		metric_type: Type of metric to compute - "entropy" or "occupancy"
	
	Returns:
		numpy.ndarray: Vector of metric values for each timepoint
	"""

	from src.helpers import calc_entropy

	kernel = load_downsampled_selection_kernel(fragment_type)	

	# Apply the kernel to select the appropriate fragment lengths
	filtered_imgs = select_w_kernel(F, kernel)
	
	# Compute the requested metric
	if metric_type.lower() == "entropy":
		# Calculate entropy along each row (timepoint)
		return np.apply_along_axis(calc_entropy, 1, filtered_imgs)
	
	elif metric_type.lower() == "occupancy":
		# Calculate mean occupancy for each timepoint
		n_timepoints = filtered_imgs.shape[0]
		return np.mean(filtered_imgs.reshape((n_timepoints, -1)), axis=1)
	
	else:
		raise ValueError(f"Unknown metric type: {metric_type}. Use 'entropy' or 'occupancy'.")


def generate_cg1_dg1_data_for_plotting(config, data, y_data=None,
	mode='ratio'):
	# Filter out non-data, incomplete deconvolutions
	# will be full of zeros
	data = data.loc[(data.sum(axis=1) > 1e-5)]

	# Calculate x and y data
	t_mean = data[config.cg1_indices()].mean(1)
	b_mean = data[config.dg1_indices()].mean(1)

	if y_data is None: 
		y_data = data.max(1)


	if mode == 'ratio':
		eps = 1e-5
		eps = 1
		x_data = np.log2((t_mean+eps) / (b_mean+eps))
	elif mode == 'difference':
		x_data = t_mean - b_mean

	plot_data = pd.DataFrame({
		'x': x_data,
		'y': y_data,
	}, index=x_data.index)

	return plot_data
