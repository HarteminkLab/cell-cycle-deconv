

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


def create_2d_gaussian_kernel(k_size=5, sigma=1, plot=False):

	# Create a grid of (x,y) coordinates
	x = np.linspace(-2, 2, k_size)
	y = np.linspace(-2, 2, k_size)
	x, y = np.meshgrid(x, y)

	# Calculate the 2D Gaussian kernel
	g_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
	g_kernel /= (2 * np.pi * sigma**2)

	# Normalize the kernel so that its sum = 1
	g_kernel /= g_kernel.sum()

	if plot:
		plt.figure(figsize=(1, 1))
		plt.imshow(g_kernel, extent=[x.min(), x.max(), y.min(), y.max()])

	return g_kernel

