
import numpy as np
import pandas as pd
from src.plot_helpers import plot_rect2
from src.global_config import GlobalConstants
from matplotlib import pyplot as plt

class ChromatinMetricTracking(object):
	"""General purpose code to track a nucleosomes position
	   based on read counts in a bin windows"""

	def __init__(self, img_data, mnase_span):
		self.img_data = img_data
		self.mnase_span = mnase_span


	def select_region(self, genomic_span, frag_lens):

		self.span = genomic_span
		self.frag_lens = frag_lens

		# Convert them to bins
		bw, bh = GlobalConstants.BIN_WIDTH, GlobalConstants.BIN_HEIGHT
		self.span_bins = genomic_span[0]//bw - self.mnase_span[0]//bw, \
			genomic_span[1]//bw - self.mnase_span[0]//bw
		self.frag_lens_bins = frag_lens[0]//bh, \
			frag_lens[1]//bh

		self.selected_span = self.span_bins[0]*bw + self.mnase_span[0],\
			self.span_bins[1]*bw + self.mnase_span[0]
		self.selected_frag_span = self.frag_lens[0]*bh, \
			self.frag_lens[1]*bh

		self.selected_img_data = self.img_data[:, 
			self.frag_lens_bins[0]:self.frag_lens_bins[1],
			self.span_bins[0]:self.span_bins[1]]


	def plot_selection(self):

		bw, bh = GlobalConstants.BIN_WIDTH, GlobalConstants.BIN_HEIGHT

		f_img_data = self.img_data
		f_mnase_span = self.mnase_span
		genomic_span_bins = self.span_bins
		frag_lens_bins = self.frag_lens_bins

		ax = plt.gca()
		ax.imshow(f_img_data[100], origin='lower', cmap='magma_r', vmin=1, vmax=10,
				  extent=[f_mnase_span[0], f_mnase_span[1], 0, GlobalConstants.MAX_Y_LEN])


		x1, x2 = genomic_span_bins[0]*bw+self.mnase_span[0], \
			genomic_span_bins[1]*bw+self.mnase_span[0]
		y1, y2 = frag_lens_bins[0]*bh, frag_lens_bins[1]*bh

		from src.plot_helpers import plot_rect2

		plot_rect2(ax, x1, y1, x2, y2, edgecolor='blue', fill=None, lw=1, zorder=100)


	def track_peak_position(self):
		from src.helpers import weighted_mean, weighted_peak_estimation
		from src.global_config import GlobalConstants

		selected_img_data = self.selected_img_data
		weighted_mean_tracking = np.zeros(selected_img_data.shape[0])

		for i in range(selected_img_data.shape[0]):
			x_positions = np.arange(self.selected_span[0], self.selected_span[1], GlobalConstants.BIN_WIDTH)
			values = selected_img_data.sum(axis=1)[i]
			weighted_mean_val = weighted_peak_estimation(x_positions, values, 100)
			weighted_mean_tracking[i] = weighted_mean_val

		self.weighted_mean_tracking = weighted_mean_tracking

	def plot_position_tracking(self, indices, ax=None):
		from src.config import load_configs_by_config_type

		config, _ = load_configs_by_config_type('shared')
		t_indices = config.get_Hpositions_for_branch('t')
		img_data = self.selected_img_data[indices]

		if ax is None:
			fig = plt.figure(figsize=(4, 3))
			ax = plt.gca()

		im = ax.imshow(img_data.sum(axis=1),

			# -5 centers the img cells on the ticks that represent the numbered values
			# -5 instead of +5 for the right extent because do not include the last
			# value in the selected span
			aspect='auto', extent=[self.selected_span[0]-5, self.selected_span[1]-5,
								   0, len(img_data)],
				  origin='lower', interpolation='none', cmap='magma_r', vmin=20, vmax=100)
		ax.plot(self.weighted_mean_tracking[indices], np.arange(len(img_data)),
				c='black', lw=1.5, ls='dotted')
