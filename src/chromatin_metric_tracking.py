
import numpy as np
import pandas as pd
from src.plot_helpers import plot_rect2
from src.global_config import GlobalConstants
from matplotlib import pyplot as plt

class ChromatinMetricTracking(object):
	"""General purpose code to track a nucleosomes position based on read counts in a bin
	windows"""

	def __init__(self, chrom_model=None, img_data=None, x_genomic_positions=None, y_fragment_length_names=None):
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


		Procedure:
		1. Load the image data from the chromatin model
		2. Define the x genomic positions and fragment lengths:
			As bp and fragment lengths, they will be spaced apart by the bin width and bin height of the deconvolution
		3. Select the range we are interested in: such as +1, -1, nucleosomal reads or small fragments around origins or promoters
			Genomic range to select, and the fragment lengths we are interested in (in genomic coordinate 
			space and fragment lengths). Note that these must be divisible by the bin width and height exactly as defined in 
			step 2.
		4. 	For nucleosome tracking as well as origin occupancy, we don't have a perfect idea of where these genomic spans lie
			so, we next need to perform a peak search and then, with a predefined search window narrow the genomic search
			space for the metric calculation.
		5. Compute the metric calculation: Currently implemented peak tracking (using a weighted mean) and occupancy changes
			in the window

		"""

		# Override with chrom model if it is provided
		if chrom_model is not None:
			self.chrom_model = chrom_model
			self.img_data = chrom_model.get_f_images()

			# Define the genomic positions and the fragment length boundaries
			# of the image data
			# Add half a bin-width to indicate that the positions are centered on the middle of the bin
			self.x_genomic_positions = np.arange(chrom_model.bin_extents[0], \
				chrom_model.bin_extents[1], GlobalConstants.BIN_WIDTH) + round(GlobalConstants.BIN_WIDTH/2)
		else:
			# Default parameters
			self.img_data = img_data.astype(float)
			self.x_genomic_positions = x_genomic_positions
			self.y_fragment_length_names = y_fragment_length_names

		# Use default y fragment lengths if none provided
		# note: used currently as we are refactoring the y fragment lengths
		if y_fragment_length_names is None:
			self.y_fragment_length_names = GlobalConstants.Y_LEN_DEFINITIONS

	def select_range(self, selected_genomic_span, selected_fragment_span):
		"""Select the genomic range and fragment lengths we are interested in."""

		selected_genomic_positions = get_genomic_positions_from_span(selected_genomic_span)
		selected_fragment_lengths = np.arange(selected_fragment_span[0], 
			selected_fragment_span[1]+GlobalConstants.BIN_HEIGHT,
		 GlobalConstants.BIN_HEIGHT)

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

	def find_peak_and_update_genomic_positions(self, window):
		"""Find peak the peak occupancy in the window (summed by time) and create a window around this peak
		to narrow the span in which we are interested in computing our metrics"""
		win_2 = window/2
		stacked_sum_data = self.selected_sum_data.sum(axis=0)
		peak = stacked_sum_data.idxmax()
		updated_span = peak-win_2, peak+win_2

		updated_genomic_positions = get_genomic_positions_from_span(updated_span)
		self.selected_sum_data = self.binned_summation_data[updated_genomic_positions]
		self.selected_genomic_positions = updated_genomic_positions


	def track_occupancy(self):
		"""Track the mass of the occupancy change"""
		self.total_occupancy = self.selected_sum_data.sum(axis=1)

	def track_entropy(self):
		"""Track the mass of the occupancy change"""
		from src.helpers import calc_entropy
		entropy_values = np.apply_along_axis(lambda row: calc_entropy(row.astype(float)), 1, 
		    self.selected_sum_data)
		self.entropy_values = pd.Series(entropy_values, self.selected_sum_data.index)

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
		img = self.img_data[100]

		extent = [self.x_genomic_positions[0], self.x_genomic_positions[-1],
			self.y_fragment_length_names[0], self.y_fragment_length_names[-1]]

		plt.imshow(img, origin='lower', cmap='magma_r', aspect='auto',
				  extent=extent)
		
		center_line = (extent[0]+extent[1])/2
		plt.axvline(center_line, c='black', lw=1, ls='dotted')

		ax = plt.gca()

		self.plot_selected_range_rect(ax)

		return fig, ax


	def plot_selected_range_rect(self, ax):
		x1, x2 = self.selected_genomic_positions[0], self.selected_genomic_positions[-1]
		y1, y2 = self.selected_fragment_lengths[0], self.selected_fragment_lengths[-1]
		plot_rect2(ax, x1, y1, x2, y2, edgecolor='blue', fill=None, lw=1, zorder=100)
		plt.xticks([])
		plt.yticks([])


	def get_center_selected_bp(self):
		gp = self.selected_genomic_positions
		gp_center = (gp[0] + gp[-1])/2.
		return gp_center


def get_genomic_positions_from_span(search_span):
	"""Create an np array from the given span"""
	# Selected genomic range and fragment lengths
	selected_genomic_positions = np.arange(*search_span, GlobalConstants.BIN_WIDTH)
	return selected_genomic_positions

# todo: may convert the search span conversion to bin positions with this method
# 
# def find_nearest(arr, num):
# 	"""For translating found values to nearest bins and subselecting columns"""
# 	arr = np.array(arr)
# 	# Filter values greater than or equal to the input number
# 	valid_values = arr[arr >= num]
# 	if valid_values.size == 0:
# 		return None  # If no values are greater or equal, return None
# 	# Find the minimum of these values
# 	nearest_value = valid_values.min()
# 	return nearest_value
