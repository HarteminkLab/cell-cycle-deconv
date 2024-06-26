
import numpy as np
import pandas as pd
from src.plot_helpers import plot_rect2
from src.global_config import GlobalConstants
from matplotlib import pyplot as plt

class ChromatinMetricTracking(object):
	"""General purpose code to track a nucleosomes position based on read counts in a bin
	windows"""

	def __init__(self, chrom_model):
		"""
		Given image data (np 3d array): (time/index x rows x columns)

		We are interested in selected a subset of the rows and columns, collapsing the rows, to get 
		a summary of a fragment length section to produce a new data frame.

		Summed occupancy reads for a fragment length span (such as nucleosome reads): 

			(time/index x columns)

		The columns should be labeled by their originally defined genomic position. Such that we will be able
		to track nucleosomal position based on the original genomic coordinate system.

		Note it is tricky working in the numpy array bin space because values need to be scaled by the bin width
		and height and translated from the np.array 0-index to the genomic space index, as np arrays do not have
		named columns nor rows.
		"""
		self.chrom_model = chrom_model
		self.img_data = chrom_model.get_f_images()

		# Define the genomic positions and the fragment length boundaries
		# of the image data
		# Add half a bin-width to indicate that the positions are centered on the middle of the bin
		self.x_genomic_positions = np.arange(chrom_model.bin_extents[0], \
			chrom_model.bin_extents[1], GlobalConstants.BIN_WIDTH) + GlobalConstants.BIN_WIDTH/2
		self.y_fragment_length_names = GlobalConstants.Y_LEN_DEFINITIONS

	def select_range(self, selected_genomic_positions, selected_fragment_lengths):

		self.selected_genomic_positions = selected_genomic_positions
		self.selected_fragment_lengths = selected_fragment_lengths
		
		# The image data is in bin space, so convert the selected fragment lengths
		# into the rows we are going to subset from the image data
		selected_fragment_bins = (selected_fragment_lengths / GlobalConstants.BIN_HEIGHT).astype(int)

		# Subset to to fragment lengths we are interested in and sum
		selected_fragment_lens_data = self.img_data[:, selected_fragment_bins]
		selected_fragment_lens_sum = selected_fragment_lens_data.sum(axis=1)

		# Create dataframe that stores the genomic position of the bin sum values
		# Each column is a timepoint, each row is the genomic position
		m = selected_fragment_lens_sum.shape[0]
		binned_summation_data = pd.DataFrame(selected_fragment_lens_sum, index=pd.Index(np.arange(m), 
			name='H_index'))
		binned_summation_data.columns = self.x_genomic_positions

		self.binned_summation_data = binned_summation_data

		# Now subset to the select genomic positions we are interested in, we have 
		# named columns now
		self.selected_sum_data = binned_summation_data[selected_genomic_positions]

	def track_occupancy(self):
		"""Track the mass of the occupancy change"""

		self.total_occupancy = self.selected_sum_data.sum(axis=1)

	def track_genomic_movement(self):
		"""Track the mass of the occupancy movement along the genomic positions"""

		# Track peaks using weighted mean
		from src.helpers import weighted_mean

		called_peak_weighted_mean = np.apply_along_axis(
			lambda row: weighted_mean(self.selected_genomic_positions, row), 1, 
			self.selected_sum_data.values)

		# Convert to a data series
		self.called_peak_weighted_mean = pd.Series(called_peak_weighted_mean, index=self.selected_sum_data.index)


	def plot_selected_region(self):
		# Plot where the self is selecting from the image data
		fig = plt.figure(figsize=(6, 1))
		img = self.img_data[5]

		extent = [self.x_genomic_positions[0], self.x_genomic_positions[-1],
			self.y_fragment_length_names[0], self.y_fragment_length_names[-1]]

		plt.imshow(img, origin='lower', cmap='magma_r', aspect='auto',
				  extent=extent)

		ax = plt.gca()

		x1, x2 = self.selected_genomic_positions[0], self.selected_genomic_positions[-1]
		y1, y2 = self.selected_fragment_lengths[0], self.selected_fragment_lengths[-1]

		plot_rect2(ax, x1, y1, x2, y2, edgecolor='blue', fill=None, lw=1, zorder=100)
		plt.xticks([])
		plt.yticks([])
		
		center_line = (extent[0]+extent[1])/2
		plt.axvline(center_line, c='black', lw=1, ls='dotted')
