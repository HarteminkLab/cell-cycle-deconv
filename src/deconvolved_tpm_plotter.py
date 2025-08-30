
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class DeconvolvedTPMPlotter:
	"""
	Class to plot the deconvolved TPM on the locus plots.

	This plotter will plot the tpm levels for transcripts in a window
	as colored (red/blue) thin boxes on the chromatin heatmaps
	"""

	def __init__(self, expression_data, transcripts):
		self.expression_data = expression_data
		self.transcripts_boundaries = transcripts

	def set_chrom_span(self, chrom, window_span):
		# Subset the transcripts to the region

		transcripts_boundaries = self.transcripts_boundaries
		self.transcripts_in_window = transcripts_boundaries[(transcripts_boundaries.chr == chrom) & 
									 (transcripts_boundaries.full_transcript_end > window_span[0]) & 
									 (transcripts_boundaries.full_transcript_start < window_span[1])]
		self.transcription_fs_in_window = self.expression_data.loc[self.transcripts_in_window.index]
		self.chrom = chrom
		self.window_span = window_span


	def plot_mean_transcripts_for_time_indices(self, ax, time_indices, vmax=16):

		# This will place the tpm lines near the bottom of a chromatin 
		# heatmap (with fragment length positions at 0, 250)
		plotting_ys = (0, 30)

		# Plotting the stranded tpm values will be centered 
		# offset positively or negatively based on the defined plotting y
		# window range
		mid_y = (plotting_ys[0]+plotting_ys[1])/2
		half_y = (plotting_ys[1]-plotting_ys[0])/2

		def plot_transcript_as_line(ax, transcript):

			tx_levels = [self.transcription_fs_in_window.loc[transcript.name].loc[time_index] for 
				time_index in time_indices]
			tx_level = np.mean(tx_levels)

			tx_span = transcript.full_transcript_start, transcript.full_transcript_end

			if transcript.strand == '+':
				ys = [mid_y, mid_y+half_y]
				cmap = plt.cm.Blues
			else:
				ys = [mid_y-half_y, mid_y]
				cmap = plt.cm.Reds

			extent = [tx_span[0], tx_span[1], ys[0], ys[1]]

			# Plot the transcript level as a heatmap (can change to use
			# vxspan)
			tx_level_mat = np.array([[tx_level]])
			ax.imshow(tx_level_mat, cmap=cmap, vmin=0, vmax=vmax, aspect='auto',
					 extent=extent, zorder=1)

		# Plot a white span background for the transcripts
		from src.plot_helpers import plot_rect2
		from src.ChromatinRNALocusPlotter import internal_spine_color
		plot_rect2(ax, self.window_span[0], plotting_ys[0], self.window_span[1], plotting_ys[1], 
			color='white', lw=0, zorder=0)

		for transcript_name, transcript in self.transcripts_in_window.iterrows():
			plot_transcript_as_line(ax, transcript)

		# Plot boundary around area designated for TPM plotting
		ax.axhline(plotting_ys[1], c=internal_spine_color, lw=1, zorder=2)
