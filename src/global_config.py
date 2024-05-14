


# Global configuration 
# for static constants


class GlobalConstants:

	BIN_WIDTH = 32
	BIN_HEIGHT = 32

	PROM_LEN = 288
	GB_LEN = 512

	MAX_Y_LEN = 256

	NUM_BINS_X = (PROM_LEN + GB_LEN) // BIN_WIDTH
	NUM_BINS_Y = (MAX_Y_LEN) // BIN_HEIGHT

	IMAGE_SHAPE = (NUM_BINS_Y, NUM_BINS_X)

	BIN_EXTENTS = [-PROM_LEN, GB_LEN, 0, MAX_Y_LEN]

	CHROM_WT1_TIMEPOINTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
	CHROM_WT2_TIMEPOINTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

	# Chromatin measures
	PLUS_ONE_FILEPATH = 'output/deconvolved_plus_one_tracking/computed_plus_one_movement_meannorm.csv'
	PLUS_ONE_METADATA_FILEPATH = 'output/deconvolved_plus_one_tracking/p1_meta_data.csv'
	GB_ENTROPY = 'output/normalized_gb_entropy.csv'
	SMALL_FRAG_OCC_FILEPATH = 'output/small_fragments_occupancy.csv'

	# Replication timing
	REPL_TIMING_FILEPATH = 'datasets/computed_mnase/all_repl_timing_deconvolved.csv'

