

# @Deprecated in lieu of simpler copy correction


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.figure_configs import FiguresConfig
from src.global_config import GlobalConstants


def load_chromatin_copy_correction(config_type, replicate):
	path = f'data/copy_correction/chromatin/copy_correction_replicate{replicate}.csv'
	correction = pd.read_csv(path).set_index(['chr', 'start'])
	return correction


def lookup_origin_copy_correction(copy_correction, origin):
	"""Look up the copy correction vector for an origin from the
	genomic copy correction table"""
	
	from src.CopyNumberCorrection import get_bin_for_position

	chrom = origin.chr
	pos = gene.pos

	start_indices = copy_correction.loc[chrom].index
	bin_idx, bin_start_bp = get_bin_for_position(pos, start_indices)

	vector = copy_correction.loc[chrom].loc[bin_start_bp]
	return vector


def lookup_copy_correction(copy_correction, chrom, pos):
	"""Look up the copy correction vector for the gene from the
	genomic copy correction table"""

	from src.CopyNumberCorrection import get_bin_for_position

	start_indices = copy_correction.loc[chrom].index
	bin_idx, bin_start_bp = get_bin_for_position(pos, start_indices)

	vector = copy_correction.loc[chrom].loc[bin_start_bp].values

	# todo: if there is an issue with the 10k window copy correction
	# the correction vector defaults to 0's. Rather we would like to 
	# interpolate from neighbors
	if vector.sum() == 0:
		print("Copy correction vector does not exist, using no correction. "
			  f"position: {pos}, chr: {chrom}, {pos}")
		vector = np.ones_like(vector)

	return vector


def lookup_gene_copy_correction(copy_correction, gene):
	"""Look up the copy correction vector for the gene from the
	genomic copy correction table"""

	from src.CopyNumberCorrection import get_bin_for_position

	chrom, pos = gene.chr, gene.TSS
	start_indices = copy_correction.loc[chrom].index
	bin_idx, bin_start_bp = get_bin_for_position(gene.TSS, start_indices)

	vector = copy_correction.loc[chrom].loc[bin_start_bp].values

	# todo: if there is an issue with the 10k window copy correction
	# the correction vector defaults to 0's. Rather we would like to 
	# interpolate from neighbors
	if vector.sum() == 0:
		print("Copy correction vector does not exist, using no correction. "
			  f"gene: {gene['gene']}, chr: {chrom}, {pos}")
		vector = np.ones_like(vector)

	return vector
