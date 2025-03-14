
import numpy as np


# Global configuration 
# for static constants


class GlobalConstants:

	BIN_WIDTH = 10
	BIN_HEIGHT = 10
	MAX_Y_LEN = 253

	# Defined fragment length boundaries
	Y_LEN_DEFINITIONS = np.arange(0, MAX_Y_LEN+BIN_HEIGHT, BIN_HEIGHT)

	# Gene definitions
	# Add one more bin to center the +1 on a bin
	# Previous run: 495, 495
	#PROM_LEN = 495*3
	#GB_LEN = 495*3
	#NUM_BINS_X = (PROM_LEN + GB_LEN + BIN_WIDTH) // BIN_WIDTH

	PROM_LEN = 495*2
	GB_LEN = 495*2
	NUM_BINS_X = (PROM_LEN + GB_LEN + BIN_WIDTH) // BIN_WIDTH
	
	# Define the promoter and gene body regions relative to a +1
	# at the zero position (oriented left to right)
	PROM_REGION = (-PROM_LEN-BIN_WIDTH/2), BIN_WIDTH/2
	GB_REGION = -BIN_WIDTH/2, GB_LEN+BIN_WIDTH/2

	NUM_BINS_Y = (MAX_Y_LEN) // BIN_HEIGHT
	IMAGE_SHAPE = (NUM_BINS_Y, NUM_BINS_X)
	BIN_EXTENTS = [-PROM_LEN-BIN_WIDTH/2, GB_LEN+BIN_WIDTH/2, 0, MAX_Y_LEN]

	# Origin definitions
	ORC_BIN_PADDING = 979
	NUM_BINS_ORC_X = (ORC_BIN_PADDING*2 + BIN_WIDTH) // BIN_WIDTH
	NUM_BINS_ORC_Y = (MAX_Y_LEN) // BIN_HEIGHT
	ORC_IMAGE_SHAPE = (NUM_BINS_ORC_Y, NUM_BINS_ORC_X)
	ORC_BIN_EXTENTS = [-ORC_BIN_PADDING-BIN_WIDTH/2, ORC_BIN_PADDING+BIN_WIDTH/2, 0, MAX_Y_LEN]

	# 70 min dropped for quality
	EXPRESSION_WT1_TIMEPOINTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
	EXPRESSION_WT2_TIMEPOINTS = [0, 10, 20, 30, 40, 50, 60, 80, 90, 100, 110, 120, 130, 140]

	CHROM_WT1_TIMEPOINTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
	CHROM_WT2_TIMEPOINTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

	# Chromatin measures
	PLUS_ONE_FILEPATH = 'output/deconvolved_plus_one_tracking/computed_plus_one_movement_meannorm.csv'
	PLUS_ONE_METADATA_FILEPATH = 'output/deconvolved_plus_one_tracking/p1_meta_data.csv'
	GB_ENTROPY = 'output/normalized_gb_entropy.csv'
	SMALL_FRAG_OCC_FILEPATH = 'output/small_fragments_occupancy.csv'

	# Replication timing
	REPL_DECONV_BIN_WIDTH = 10000
	REPL_DECONV_BIN_STEP = 2000


def load_expression_timepoints(replicate):
	if replicate == 1:
		return GlobalConstants.EXPRESSION_WT1_TIMEPOINTS
	else:
		return GlobalConstants.EXPRESSION_WT2_TIMEPOINTS

def load_chrom_timepoints(replicate):
	if replicate == 1:
		return GlobalConstants.CHROM_WT1_TIMEPOINTS
	else:
		return GlobalConstants.CHROM_WT2_TIMEPOINTS



def fragment_lengths_definitions():
	"""Length spans as defined from the replicate 2 dataset,
	these should also match replicate 1


	Prefer using the kernel selection method, see chromatin_metrics.py

	"""
	nucleosome_len_span=(144, 192)
	mid_frag_span=(96, 144)
	small_frag_span=(0, 96)
	return small_frag_span, mid_frag_span, nucleosome_len_span


def len_bins():
	"""
	Defined locations for small, intermediate and nucleosome size bins
	"""

	small_lens, med_lens, nuc_lens = fragment_lengths_definitions()
		
	small_bins = small_lens[0]//GlobalConstants.BIN_HEIGHT,  \
		small_lens[1]//GlobalConstants.BIN_HEIGHT
	med_bins = med_lens[0]//GlobalConstants.BIN_HEIGHT, \
		med_lens[1]//GlobalConstants.BIN_HEIGHT
	nuc_bins = nuc_lens[0]//GlobalConstants.BIN_HEIGHT, \
		nuc_lens[1]//GlobalConstants.BIN_HEIGHT

	return small_bins, med_bins, nuc_bins
