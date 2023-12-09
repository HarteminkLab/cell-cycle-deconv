
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from cc_src.sgd import get_gene_name_orf_name
from cc_src.mnase_plotting import plot_mnase_density


class ChromatinGrid:
	"""
	In this class, we will be taking mnase-seq reads for a gene, and computing a grid of occupancy 
	values for the gene's locus.

	That we can then deconvolve.

	We will be able to plot the gene's locus (raw data) as well as the histogram of the grid for 
	verification that the grid values
	are created properly.

	Notes: We will eventually need to normalize or scale (by copy number and/or by sample depth)
	"""


	def __init__(self):

		self.padding = 1000
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')

	def set_gene(self, gene_name):

		# Get some gene information
		self.orf_name, self.gene_name = get_gene_name_orf_name(gene_name)
		self.gene = self.geneset.loc[self.orf_name]

		self.chr_reads = pd.read_hdf(f'output/mnase/yl_rep2_mnase_reads/yl_rep2_mnase_reads_chr{self.gene.chr}.h5', 
					'mnase_data')

		self.mnase_span = self.gene.TSS-self.padding, self.gene.TSS+self.padding

		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		self.times = self.gene_reads['sample'].unique()

	def compute_bin_counts_sample(self, sample):

		plotting_reads = self.gene_reads[self.gene_reads['sample'] == sample]

		x_bin_size = 100
		y_bin_size = 50

		xlims = self.mnase_span
		x_bins = np.arange(xlims[0], xlims[1]+x_bin_size, x_bin_size)
		y_bins = np.arange(50, 200+y_bin_size, y_bin_size)

		hist, x_edges, y_edges = np.histogram2d(plotting_reads['mid'], 
			plotting_reads['length'], bins=[x_bins, y_bins])

		return plotting_reads, hist, x_edges, y_edges

	def create_bins_per_sample(self):

		samples = self.gene_reads['sample'].unique()

		self.all_plotting_reads = {}
		self.all_hists = {}
		self.all_x_edges = {}
		self.all_y_edges = {}

		for sample in samples:
			plotting_reads, hist, x_edges, y_edges = self.compute_bin_counts_sample(sample)

			self.all_plotting_reads[sample] = plotting_reads
			self.all_hists[sample] = hist.T
			self.all_x_edges[sample] = x_edges
			self.all_y_edges[sample] = y_edges

		hist = self.all_hists[0]

		print(f"The histogram shape for the 2000 bp window around the TSS:", 
			hist.shape)

		# If we were to take the middle 10 bins (equivalent to 1000 bp window around the TSS), 
		# Our histogram for this time point would look like:
		print("The shape for the middle 10 bins (1000 bp around the TSS):", 
			hist[:, 5:-5].shape)


	def plot_sample(self, ax1, ax2, sample):
		
		xlims = self.mnase_span
		gene = self.gene

		plotting_reads = self.all_plotting_reads[sample]
		hist = self.all_hists[sample]
		x_edges = self.all_x_edges[sample]
		y_edges = self.all_y_edges[sample]

		plot_mnase_density(ax1, plotting_reads)
		ax1.set_xticks([])
		ax1.set_xlim(*xlims)

		ax2.imshow(hist, origin='lower', aspect='auto', cmap='magma_r',
			extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
		ax2.set_xlim(*xlims)

		for ax in [ax1, ax2]:
			for x in [gene.TSS-500, gene.TSS, gene.TSS+500]:
				ax.axvline(x, c='green', alpha=0.75, lw=3)
				
			ax.set_ylim(50, 200)

			xticks = np.arange(gene.TSS-1000, gene.TSS+1500, 500)
			xtick_labels = ['-1000', '-500', 'TSS', '500', '1000']

			ax.set_xticks(xticks)
			ax.set_xticklabels(xtick_labels)


	def plot_raw_and_grid(self):

		times = self.times
		fig, axs = plt.subplots(len(times), 2, figsize=(19, 19))

		# Get a list of the raw and grid axes by transposing the axes
		axs = np.array(axs).T
		raw_axes = axs[0]
		grid_axes = axs[1]

		# Plot the raw data and the grid histograms for each of the time points
		for i in range(len(times)):
			time = times[i]
			raw_ax = raw_axes[i]
			grid_ax = grid_axes[i]
			self.plot_sample(raw_ax, grid_ax, time)

	def create_deconvolution_matrices(self, plot=False):

		hist = self.all_hists[0]
		times = self.times

		# So let's create the data structure for inner 1000 bp histogram for all time points:
		# Dimension: (num_time_points, rows, columns)

		inner_hist_shape = hist[:, 5:-5].shape
		threed_hist_matrix = np.zeros((len(times), inner_hist_shape[0], inner_hist_shape[1]))

		for i in range(len(times)):
			time = times[i]
			inner_hist = self.all_hists[time][:, 5:-5]
			threed_hist_matrix[i] = inner_hist

		self.threed_hist_matrix = threed_hist_matrix

		# Checking if we can collapse the rows and columns, then restore them
		reshaped_hist = threed_hist_matrix.reshape(15, -1)
		first_hist = reshaped_hist[0]

		if plot:
			print("The first histogram is shape:", threed_hist_matrix[0].shape)
			print("Reshaping this histogram to a vector of shape:", first_hist.shape)
			restored_hist = first_hist.reshape((3, 10))
			print("Then, if we were to take that first vector and restore it to its original shape:", 
				  restored_hist.shape)

			plt.subplot(1, 2, 1)
			plt.imshow(threed_hist_matrix[0], origin='lower', cmap='magma_r')
			plt.title("Original first histogram")
			plt.xticks([])
			plt.yticks([])


			plt.subplot(1, 2, 2)
			plt.imshow(restored_hist, origin='lower', cmap='magma_r')
			plt.title("Restored histogram after flattening")
			plt.xticks([])
			plt.yticks([])

		# We will need to drop the 110 time point for this deconvolution
		# TODO: At least for now, as we have assumed we should drop this point as per 
		# Yulong's analysis
		# We probably don't need to do this anymore.
		deconv_hist = np.concatenate([reshaped_hist[:-4, :], reshaped_hist[-3:, :]])
		print("Now we have a data structure that we can try to deconvolve of shape:", 
			  deconv_hist.shape)

		# Reshape for deconvolution
		self.deconv_hist = deconv_hist

	def create_deconvolution_plots(self, f, model):
		from src.model import color_for_key

		reshaped_f = f.reshape(-1, 3, 10)
		phase_cols = model.config.phase_columns

		phases = []
		indices = []

		for key, values in phase_cols.items():
			phases = phases + [key] * len(values)
			indices = indices + list(values)

		phase_col_df = pd.DataFrame({'phase': phases, 'column': indices})
		phase_col_df = phase_col_df.set_index('column')
		phase_col_df.head()

		def color_for_index(phase_col_df, index):
			phase = phase_col_df.loc[index].phase
			color = color_for_key(phase)
			return color

		fig, axs = plt.subplots(26, 10, figsize=(13, 9))
		axs = np.array(axs).T.flatten()

		plotting_index = 0
		last_phase = None
		for i in range(len(axs)):
			
			ax = axs[plotting_index]

			if plotting_index >= reshaped_f.shape[0]: 
				ax.set_xticks([])
				ax.set_yticks([])
				continue

			phase = phase_col_df.loc[i].phase
			color = color_for_index(phase_col_df, i)

			im = ax.imshow(reshaped_f[i], origin='lower', cmap='magma_r', aspect='auto', vmax=500)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.axvline(4.5, c=color, lw=2)

			if last_phase is not None and phase != last_phase:
				plotting_index += 2
			else:
				plotting_index += 1

			last_phase = phase

	def plot_prediction_comparison(self, model, f):
		times = self.times
		predicted_g = np.matmul(model.H, f)

		n = predicted_g.shape[0]

		predicted_g_reshaped = predicted_g.reshape(n, 3, 10)

		fig, axs = plt.subplots(n, 2, figsize=(4, 6))
		axs = np.array(axs).T
		g_axs = axs[0]
		pred_g_axs = axs[1]

		for i in range(n):
			time = times[i]

			g_ax = g_axs[i]
			im = g_ax.imshow(self.threed_hist_matrix[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=300)
			
			pred_ax = pred_g_axs[i]
			im = pred_ax.imshow(predicted_g_reshaped[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=300)
			
			for ax in [g_ax, pred_ax]:
				ax.set_xticks([])
				ax.set_yticks([])
				ax.axvline(4.5, c='black', lw=2)

		g_axs[0].set_title("Original")
		pred_g_axs[0].set_title("Predicted")