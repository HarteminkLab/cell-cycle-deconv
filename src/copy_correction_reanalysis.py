
import numpy as np


class CopyCorrectionAnalysis:
	"""In this class, copy correction is revisited. The idea is to simplify the copy
	number correction process and clarify step-by-step how the process is performed:

	Procedure:
	1. Retrieve the 10 kb occupancy counts for all chromosomes
	2. Normalize these occupancy counts to match up the first cell cycle min and max 
		occupancy values. 
		Assumption: All segments of the genome will reach these same minimum and maximal values
			Reality: may not be the case as cells begin entering G1 by late S/G2M, 
			the maximal value is lowered and mixed with G1 cells.
	3. Retrieve each of these 10 kb occupancy values for each gene.
	4. Sort by replication timing
	5. Retrieve the estimated replication timing index from the deconvolution run.
	4. Compute an estimated copy number for each timepoint for each gene. This is computed
		from H (CLOCCS predictions of mixture of G1,S,G2/M at each timepoint with granularity
		for replication timing index to indicate when replication occurs).
	5. Correct for copy number. Assuming normalization and scaling is sufficient, compute the
		copy corrected occupancy counts.

	Expectation:
	1. Occupancy values that peak during S phase should now be reduced such that the
		approximate changes in occupancy should be around 1.0.

	Notes to address:
	- Mind the normalization schemes for the copy number proportions computed from H
	- The normalization for the 10 kb occupancy windows:
		computed as min and maximal values in the first cell cycle time frame
	"""
	def __init__(self):

		from src.mnase_replication_timing_analysis import MNaseOriginAnalysis

		# Compute the 10kb occupancies per the genome
		mnase_occupancies = MNaseOriginAnalysis(replicate=1)
		mnase_occupancies.compute_bin_curves()
		mnase_occupancies.normalize_and_compute_raw_replication_timing()
		self.mnase_occupancies_1 = mnase_occupancies

		mnase_occupancies = MNaseOriginAnalysis(replicate=2)
		mnase_occupancies.compute_bin_curves()
		mnase_occupancies.normalize_and_compute_raw_replication_timing()
		self.mnase_occupancies_2 = mnase_occupancies

		# Compute H for replicate 1 and 2
		from src.config import load_configs_by_config_type
		config1, config2 = load_configs_by_config_type('shared')
		self.H1, Hpos = config1.calcH_function(config1.intervals_wt1, config1.WT1_TIMEPOINTS)
		self.H2, Hpos = config2.calcH_function(config2.intervals_wt1, config2.WT1_TIMEPOINTS)


def create_copy_number_H(H, replication_idx):
    """Create a copy number matrix from H, converting indices from the replication index
    onward to two copies.
    
    The resulting matrix is a modification of the original proportion matrix that represents
    the overall expected copy number per timepoint when the columns are collapsed
    """
    c1_indices = np.concatenate([np.arange(replication_idx), np.array([H.shape[1]-1])])
    c2_indices = np.arange(replication_idx, H.shape[1]-1)

    # Combine the two for the expected copy number for the gene
    H_expected_copy_num = H.copy()
    H_expected_copy_num[:, c2_indices] = H[:, c2_indices]*2
    return H_expected_copy_num


def correct_replication_indices(H, replication_indices):
    """Create a combined H matrix that includes each of the copy number 
    corrected H matrices.
    
    Then create a normalized copy number matrix per gene. A matrix that represents
    the copy correction including the normalizing effect of varying replication times
    per genome segment.
    """
    num_genes = len(replication_indices)
    H_expected_copy_per_gene = np.zeros((num_genes, *H.shape))

    for i in range(num_genes):
         H_expected_copy_per_gene[i] = create_copy_number_H(H, replication_indices[i])

    H_combined_copy_num = np.sum(H_expected_copy_per_gene, axis=0) / num_genes
    overall_sum = H_combined_copy_num.sum(axis=1)
    H_normalized_copy_per_gene = H_expected_copy_per_gene.sum(axis=2).T / \
        overall_sum.reshape((-1, 1))
    return H_combined_copy_num, H_expected_copy_per_gene, H_normalized_copy_per_gene


