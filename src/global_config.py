


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
