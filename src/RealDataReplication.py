
import numpy as np
import cvxpy as cp
from src.timer import Timer
from matplotlib import pyplot as plt
from src.replication_deconvolution_solver import deconvolve_avg_copy_curve, \
	estimate_rough_average_copy_curve_fit, deconvolve_replication


class RealDataReplicationDeconvolution():
	"""This model deconvolve the replication timing.
	"""
	def __init__(self, config, chr=10, replicate=1):

		self.load_replicate_data(chr, replicate)
		self.setup_deconvolution(config)


	def load_replicate_data(self, chr, replicate):

		from src.mnase_replication_timing_analysis import MNaseOriginAnalysis

		mnase_analysis_rep = MNaseOriginAnalysis()
		mnase_analysis_rep.load_mnase_data(replicate=replicate, chromosome=chr)

		mnase_analysis_rep.compute_sliding_window_counts_all_times()
		unnormalized_total_occupancy = mnase_analysis_rep.all_counts_unnormalized_df
		normalized_total_occ = unnormalized_total_occupancy / \
			unnormalized_total_occupancy.mean(axis=0).values.reshape((1, -1))

		self.mnase_analysis_rep = mnase_analysis_rep
		self.unnormalized_total_occupancy = unnormalized_total_occupancy
		self.normalized_occupancy = normalized_total_occ
	
		# Setup regions to threshold, 
		# regions with low occupancy will be omitted when needed
		self.selected_threshold_region = self.normalized_occupancy.T.mean(axis=0) > 0.75


	def setup_deconvolution(self, config):
		"""Setup the deconvolution:

		1. The config and H
		2. The data to deconvolve, transformed via negative binomial
		"""

		self.config = config
		self.H, _ = config.calcH_function(config.intervals_wt1, 
			config.WT1_TIMEPOINTS)

		# Data setup: 
		from src.transformations import log_transform_counts

		# Transform the raw data, scale values by 100 to resemble
		# counts data, unsure of this is necessary at this point.
		# But we have been working copy number like values up to here
		# so we want to make them more resemble read counts.
		all_transformed_data = log_transform_counts(
			self.normalized_occupancy.values*100)

		# Mean transformation
		normalized_transformed_data = (all_transformed_data / \
			all_transformed_data.mean(axis=0))

		self.normalized_transformed_data = normalized_transformed_data


	def plot_rough_copy_curve_estimation(self):
		plt.figure(figsize=(4, 2))
		plt.plot(self.selected_copy_curves[:, :].T)
		plt.plot(self.rough_average_copy_curve, lw=10)
		plt.title("Initial/rough copy curve estimation")


	def estimate_rough_average_copy_curve_fit(self):
		"""The replication deconvolution needs an estimate for the average
		copy curve. First pass can be a sample of 10 sites to fit
		for this curve by minimizing the rn for all possible values of 
		replication timing. Each fit will be very rough, but the average
		copy number curve should be a pretty good estimate"""

		from src.helpers import normalize_max_min
		from src.timer import Timer

		H = self.H
		config = self.config
		G = self.normalized_transformed_data.T

		self.all_avg_copy_curves, self.found_replication_indices, \
		self.selected_copy_curves, self.rough_average_copy_curve \
			= estimate_rough_average_copy_curve_fit(config, H, G)

	def deconvolve(self):
		"""
		Deconvolve the replication curve from the H, G and average copies curve
		"""

		config = self.config
		H = self.H
		G = self.normalized_transformed_data.T
		avg_copies_per_time = self.rough_average_copy_curve

		self.F = deconvolve_replication(config, H, G, avg_copies_per_time)


	def plot_replication_deconvolution(self, fig=None):
		if fig is None:
			fig = plt.figure(figsize=(7, 1))

		plt.imshow(self.F, origin='lower', aspect='auto')
		plt.ylim(40, 60)
		plt.xticks([])


	def plot_deconvolution_prediction(self):

		from src.RealDataReplication import threshold_selection

		new_replication_mat = self.F
		rough_average_copy_curve = self.rough_average_copy_curve
		H = self.H

		cmap = plt.get_cmap('RdBu_r')
		cmap.set_bad('#111', 1.)

		predicted_G = H@((self.F+1)/\
		    self.rough_average_copy_curve.reshape((-1, 1)))
		                 
		raw_data = self.normalized_occupancy.T
		G = self.normalized_transformed_data.T

		thresholded_raw_data = threshold_selection(raw_data.values, 
		    self.selected_threshold_region, fill=np.nan, renormalize=True)

		thresholded_G = threshold_selection(G, 
		    self.selected_threshold_region,
		                   fill=np.nan, renormalize=True)

		thresholded_predicted_G = threshold_selection(predicted_G, 
		    self.selected_threshold_region,
		                   fill=np.nan, renormalize=True)

		# To start, here is what the data will look like when 
		# we take our existing replication
		# curve with our average copy curve.
		fig = plt.figure(figsize=(7, 7))
		plt.subplots_adjust(hspace=0.5)

		plt.subplot(5, 1, 1)
		self.plot_replication_deconvolution(fig=fig)
		plt.xticks([])
		plt.colorbar()
		plt.title("Deconvolved replication timing")

		plt.subplot(5, 1, 2)
		plt.imshow(thresholded_raw_data, cmap=cmap, vmin=0, vmax=2,
          origin='lower', aspect='auto')
		plt.title("Raw data, untransformed")
		plt.xticks([])
		plt.colorbar()

		plt.subplot(5, 1, 3)
		plt.imshow(thresholded_G, aspect='auto', origin='lower', 
			cmap=cmap,
			vmin=0.75, vmax=1.25)
		plt.colorbar()
		plt.xticks([])
		plt.title("G: Raw data log-transformed")

		plt.subplot(5, 1, 4)
		plt.imshow(thresholded_predicted_G, aspect='auto', origin='lower', 
			cmap=cmap, vmin=0.75, vmax=1.25)
		plt.colorbar()
		plt.xticks([])
		plt.title("Prediction")

		residual_G = thresholded_G - thresholded_predicted_G
		plt.subplot(5, 1, 5)
		plt.imshow(residual_G, aspect='auto', origin='lower', 
			cmap=cmap,
			vmin=-0.25, vmax=0.25)
		plt.colorbar()
		plt.xticks([])
		plt.title("Residual: G-prediction")

		plt.subplots_adjust(hspace=0.5)

	def plot_nb_data(self):

		plt.figure(figsize=(7, 1))

		plt.imshow(thresholded_nb_data, vmin=0.75, vmax=1.25,
				   cmap='RdBu_r', aspect='auto',
				  origin='lower')
		plt.xticks([])
		plt.title("Raw data, Negative binomial transformation")
		plt.colorbar()


def threshold_selection(dat, threshold_selection, fill=1.,
	renormalize=False):
	new_dat = dat.copy()
	new_dat[:, ~threshold_selection] = fill

	# If thresholded to fill with nans, we can renormalize such
	# that the non-thresholded out regions mean to 1
	if renormalize:

		# Get the current mean of the good rows
		row_means = np.nanmean(new_dat, axis=1)

		# Divide the data by these mean
		new_dat = new_dat / row_means.reshape((-1, 1))

	return new_dat
