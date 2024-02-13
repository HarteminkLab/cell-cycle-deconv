

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


def load_scaling_mat(replicate):
	import pandas as pd
	len_scaling_mat = pd.read_csv('datasets/computed_mnase/combined_len_scaling.csv').set_index('length')
	rep_cols = len_scaling_mat.columns[len_scaling_mat.columns.str.endswith(f'_replicate{replicate}')]
	rep_scaling_mat = len_scaling_mat[rep_cols]
	timepoints = [int(c) for c in rep_cols.str.replace(f'_replicate{replicate}', '')]
	rep_scaling_mat.columns = timepoints
	return rep_scaling_mat


	