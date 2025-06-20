
import numpy as np
from scipy.stats.distributions import norm

def get_smoothing_kernel(plot=False, smoothing_window=200, smoothing_sigma=10):
	n = smoothing_window
	xs = np.linspace(-n//2, n//2+1, n)
	kernel = norm.pdf(xs, 0, smoothing_sigma)
	kernel = kernel/kernel.sum()

	if plot:
		import matplotlib.pyplot as plt
		plt.plot(xs, kernel)

	return kernel

def smooth_rna_curve(input_pileup):
	"""
	Apply Gaussian smoothing to RNA pileup curve.
	Adapted from notebook's smooth_rna_curve function.
	"""
	# Create Gaussian kernel
	kernel = get_smoothing_kernel()
	
	# Apply convolution
	smoothed_pileup = np.convolve(input_pileup, kernel, mode='same')
	return smoothed_pileup
