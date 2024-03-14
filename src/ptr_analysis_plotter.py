

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class PTRAnalysisPlotter:
	"""Class to plot example peak-to-trough ratios for different genes sorted
	by various metrics.

	As well as for plotting individual thresholds.
	"""

	def __init__(self, peak_to_trough_analysis, metric_values):
		self.peak_to_trough_analysis = peak_to_trough_analysis
		self.metric_values = metric_values

	def set_quantiles(self, qs):
		self.qs = qs


	def plot_distribution(self):

		plt.figure(figsize=(3, 2))
		plt.hist(self.metric_values, bins=100)

		q_vals = np.quantile(self.metric_values, self.qs)

		for q_val in q_vals:
			plt.axvline(q_val, zorder=0, c='#fdd', lw=1)


	# A function to order by the metric, and plot 20 genes equally spaced along the distribution of the metric
	def plot_metric_examples_quantiles(self, low=1, hi=5, vmax=10):

		peak_to_trough_analysis = self.peak_to_trough_analysis
		ptrs = peak_to_trough_analysis.ptrs_df
		vals = self.metric_values
		qs = self.qs

		# Now we have ORFs for our quantiles we can compare the images for each
		# of these quantile steps
		quantile_orf_names = indices_for_quantiles(vals, self.qs)

		n = len(quantile_orf_names)
		fig, axs = plt.subplots(n, 2, figsize=(3.5, 3./5. * n))
		plt.subplots_adjust(hspace=0.5, top=0.8)
		
		axs = np.array(axs).T
		raw_axs = axs[0]
		thresh_axs = axs[1]
		
		for i in range(n):

			orf_name = quantile_orf_names[i]
			quantile_val = qs[i]
			title = f"perc={quantile_val*100:.1f}% - {vals.loc[orf_name]:.2f}\n{orf_name}"
			ax = raw_axs[i]
			ax.set_ylabel(title, rotation=0, ha='right')
			
			strand = peak_to_trough_analysis.geneset.loc[orf_name].strand

			img = ptrs.loc[orf_name].values.reshape(peak_to_trough_analysis.image_shape).astype(float)        
			
			def plt_img(ax, img):

				if strand == '-':
					img = np.flip(img, axis=1)

				ax.imshow(img, cmap='viridis', vmax=vmax, origin='lower', aspect='auto',
						 extent=[0, img.shape[1], 0, img.shape[0]])

				# todo: hard-coded bin dimensions
				bin_width = 32
				prom_size = 288
				window_size = 800
				ax.set_xticks([])
				ax.set_yticks([])

				ax.axvline(prom_size/bin_width, c='white', alpha=0.25)
					
			plt_img(ax, img)

			thresholded_image = threshold_img(img, L=low, H=hi)
			
			plt_img(thresh_axs[i], thresholded_image*vmax)


	def plot_threshold_scan(self, gene_name):

		peak_to_trough_analysis = self.peak_to_trough_analysis
		geneset = peak_to_trough_analysis.geneset
		gene = geneset[geneset.gene == gene_name].iloc[0]
		orf_name = gene.name
		gene_ptr = peak_to_trough_analysis.ptrs_df.loc[orf_name]
		gene_ptr_img = gene_ptr.values.reshape(peak_to_trough_analysis.image_shape).astype(float)

		hs = np.linspace(4, 6, 4)
		ls = np.linspace(1, 2, 4)
		num_h = len(hs)
		num_l = len(ls)

		plt.figure(figsize=(1.5, 0.75))

		plt.imshow(gene_ptr_img, cmap='Blues', origin='lower', aspect='auto')
		plt.xticks([])
		plt.yticks([])
				
		if gene.strand == '-':
			plt.xlim(plt.xlim()[1], plt.xlim()[0])

		fig, axs = plt.subplots(num_h, num_l, figsize=(4, 3))
		plt.subplots_adjust(hspace=0.75, wspace=0.2)
		
		for i in range(num_h):
			for j in range(num_l):
				ax = axs[i][j]
				low = ls[j]
				hi = hs[i]

				thresholded_ptr = threshold_img(gene_ptr_img, L=low, H=hi)
				ax.imshow(thresholded_ptr, cmap='Blues', origin='lower', aspect='auto')
				ax.set_xticks([])
				ax.set_yticks([])
				ax.set_title(f"Hi: {hi:.1f}, Lo: {low:.1f}", fontsize=6)
				
				if gene.strand == '-':
					ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])



def threshold_img(example_img, L=1, H=6, ret_all=False):
	"""Threshold a 2D matrix/img by a low and high filter. Keep low value iff adjacent to
	a high location."""
	
	data = example_img

	# Apply thresholds
	L_mask = data > L
	H_mask = data > H

	# 2D Convolution to spread the influence of H elements
	kernel = np.ones((3, 3))  # Simple 3x3 kernel for demonstration
	H_convolved = convolve2d(H_mask.astype(int), kernel, mode='same') > 0

	# Combine H_convolved with L_mask to refine the selection
	final_mask = H_convolved & L_mask
	
	if ret_all:
		return final_mask, L_mask, H_mask, H_convolved

	return final_mask


def indices_for_quantiles(series, qs):
	"""Get a list of indices for a given list of quantiles"""
	def idxquantile(s, q):
		"""Get the index name for the given quantile"""
		qv = s.quantile(q)
		return (s.sort_values()[::-1] <= qv).idxmax()
	return [idxquantile(series, q=q) for q in qs]
