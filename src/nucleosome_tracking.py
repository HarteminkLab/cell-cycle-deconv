
from src.global_config import GlobalConstants
import numpy as np
import pandas as pd


class NucleosomeTracking(object):
	"""General purpose code to track a nucleosomes position based on read counts in a bin
	windows"""

	def __init__(self, img_data, x_genomic_positions, y_fragment_length_names, 
			selected_genomic_positions, selected_fragment_lengths):
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

		self.img_data = img_data
		
		# The image data is in bin space, so convert the selected fragment lengths
		# into the rows we are going to subset from the image data
		selected_fragment_bins = (selected_fragment_lengths / GlobalConstants.BIN_HEIGHT).astype(int)

		# Subset to to fragment lengths we are interested in and sum
		selected_fragment_lens_data = img_data[:, selected_fragment_bins]
		selected_fragment_lens_sum = selected_fragment_lens_data.sum(axis=1)

		# Create dataframe that stores the genomic position of the bin sum values
		# Each column is a timepoint, each row is the genomic position
		m = selected_fragment_lens_sum.shape[0]
		binned_summation_data = pd.DataFrame(selected_fragment_lens_sum, index=pd.Index(np.arange(m), 
		    name='H_index'))
		binned_summation_data.columns = x_genomic_positions

		self.binned_summation_data = binned_summation_data

		# Now subset to the select genomic positions we are interested in, we have 
		# named columns now
		self.selected_sum_data = binned_summation_data[selected_genomic_positions]

		# Track peaks using weighted mean
		from src.helpers import weighted_mean
		self.called_peak_weighted_mean = np.apply_along_axis(
		    lambda row: weighted_mean(selected_genomic_positions, row), 1, 
		    self.selected_sum_data.values)


def testing_new_tracker(chrom_model, p1_search_span=(110, 280)):
	"""
	Identify the p1 search span and track the location of the nucleosome mass
	across time using a weighted average
	"""

	# Identify the downstream nucleosome bin at post g1 for reference
	f_imgs = chrom_model.get_f_images()

	center_of_window = (chrom_model.bin_extents[0]+chrom_model.bin_extents[1])/2.

	if chrom_model.origin.strand == '+':
		p1_span = center_of_window+p1_search_span[0], \
			center_of_window+p1_search_span[1]
	else:
		p1_span = center_of_window-p1_search_span[1], \
			center_of_window-p1_search_span[0]

	p1_span = int(p1_span[0]), int(p1_span[1])

	# Select the nucleosome bins
	from src.chromatin_metrics import len_bins
	_, _, nuc_lens = len_bins()

	# Additional padding for nucleosome tracking
	nuc_lens = nuc_lens[0], nuc_lens[1]
	nuc_bins = f_imgs[:, nuc_lens[0]:nuc_lens[1]]
	nuc_bins_sum = nuc_bins.sum(axis=1)

	from src.helpers import weighted_mean

	# Create dataframe that stores the genomic position of the bin sum values
	# Each column is a timepoint, each row is the genomic position
	x_bin_starts = np.arange(chrom_model.bin_extents[0], chrom_model.bin_extents[1], 24)
	nuc_bin_sums = pd.DataFrame(nuc_bins_sum.T, index=pd.Index(x_bin_starts, 
	    name='bin_start'))
	p1_nuc_bin_sums = nuc_bin_sums.loc[p1_span[0]:p1_span[1]].copy()

	# Let's start with a plot of the p1 nucleosome mass over time
	t_indices = chrom_model.config.get_Hpositions_for_branch('t')

	xs = p1_nuc_bin_sums.index
	from src.helpers import weighted_mean
	p1_nuc_calling_wm = np.apply_along_axis(
	    lambda row: weighted_mean(xs, row), 0, 
	    p1_nuc_bin_sums.values)

	p1_nuc_calling_wm = pd.Series(p1_nuc_calling_wm, index=p1_nuc_bin_sums.columns)

	return p1_nuc_bin_sums, p1_nuc_calling_wm, p1_span
