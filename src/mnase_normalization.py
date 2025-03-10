
import numpy as np
from src.helpers import downsample_bins
import pandas as pd
import matplotlib.pyplot as plt


def normalize_to_target_distribution(data, target_distribution):
	"""
	Normalize the data array to match the target fragment length distribution.
	"""

	current_distributions = data.sum(axis=2)  # shape: (16, 251)
	
	epsilon = 1e-10

	# Calculate scaling factors by broadcasting target distributions along the time axis
	# Target distribution will be: (251, 1)
	# Divide by [16, 251, 1]
	target_distribution_reshaped = target_distribution.T[None, :] 

	denom = ((current_distributions[:, :]) + epsilon)
	scaling_factors = (target_distribution_reshaped /denom )[:, :, None]
	
	# Apply scaling factors to the data
	normalized_data = data * scaling_factors
	
	# Mean normalize to 1.0
	return normalized_data / normalized_data.mean()

def normalize_to_target_sums(data, target_sums):
	"""
	Normalize the data array to match target sum for each timepoint.
	"""
	# Calculate current sum for each timepoint
	current_sums = data.sum(axis=(1, 2))  # shape: (16,)
	
	# Calculate scaling factor for each timepoint
	# Add small epsilon to avoid division by zero
	epsilon = 1e-10
	scaling_factors = target_sums / (current_sums + epsilon)  # shape: (16,)
	
	# Reshape for broadcasting
	scaling_factors = scaling_factors.reshape(target_sums.shape[0], 1, 1)
	
	# Apply scaling to maintain the relative proportions within each timepoint
	# while scaling to the target sum
	normalized_data = data * scaling_factors
	
	normalized_data = normalized_data / normalized_data.mean() * target_sums.mean()
		
	return normalized_data 


def normalize_and_downsample(exact_bins, target_distribution, g):
	normalized_1 = exact_bins / exact_bins.mean(axis=1).mean(axis=1)[:, None, None]

	# Normalize to target length distribution
	length_normalized = normalize_to_target_distribution(normalized_1, 
		target_distribution.values)

	# Normalize to target sums, preserve the 10kb curves (we expect copy number variation)
	# that we will correct for
	length_normalized_target_sums = normalize_to_target_sums(length_normalized, g.values)

	# Downsample to deconvolution size
	downsampled_bins = downsample_bins(length_normalized_target_sums)
	return normalized_1, length_normalized, length_normalized_target_sums, downsampled_bins

def plot_normalization_sanity(normalized_1, length_normalized, length_normalized_target_sums, 
	downsampled_bins, target_distribution, g, axs=None):

	if axs is None:
		fig, axs = plt.subplots(3, 1, figsize=(13, 3))
		plt.subplots_adjust(top=0.8)
		plt.suptitle("Normalization verification")

	ax = axs[0]
	ax.plot(normalized_1.mean(axis=2).T, alpha=0.05, color='blue')
	ax.plot(normalized_1.mean(axis=2).T[0], alpha=0.05, color='blue', label="Raw")
	ax.plot(target_distribution, label="Target", color='red')
	ax.plot(length_normalized.mean(axis=2).T, color='black', ls='dotted')
	ax.plot(length_normalized.mean(axis=2)[0])
	plt.legend()
	ax.set_title("Length normalization")

	ax = axs[1]
	ax.plot(g.index, g.values, label="Original sums")
	ax.plot(g.index, length_normalized_target_sums.mean((1,2)), label="Length normalized sums")
	ax.plot(g.index, downsampled_bins.mean((1,2)), label="Downsampled sums")
	ax.set_title("Per sample normalizaiton")
	plt.legend()

	ax = axs[2]
	ax.plot(np.arange(0, 260, 10), downsampled_bins.mean(axis=2).T, color='black', alpha=0.1)
	ax.set_title("Post-normalizatio+downsamplng\nlength distributions")


def load_target_distribution():
	path = 'datasets/computed_mnase/target_length_distribution.csv'
	print("Loading target length distribution: ", path)
	target_distribution = pd.read_csv(path).set_index('fragment_length')
	return target_distribution['combined']
