
import numpy as np
import cvxpy as cp
from src.timer import Timer
from matplotlib import pyplot as plt


# Code to transition into using the raw data from the toy example, 
# at this point we will need to know how to handle non-synchronous populations
# and the conversion from the experimental timepoints to the single
# cell timepoints....


def deconvolve_avg_copy_curve(g, config, H, replication_index):

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')

	n, m = H.shape

	# Let's try the DCP error thing again, division by a variable is not allowed
	# but we can try to multiply by a variable....
	solver = cp.MOSEK

	# Create f from replication index
	f = np.ones(m)
	f[replication_index:] = 2

	inv_learn_avg_copies = cp.Variable(m)

	normalized_f = cp.multiply((f), inv_learn_avg_copies)
	predicted_g = H@normalized_f

	elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1

	objective = cp.Minimize(
		cp.sum(cp.norm(elementwise_result, 'fro')**2)
	)
	constraints = [inv_learn_avg_copies >=0.5, inv_learn_avg_copies <=1]

	# Enforce average copy number of 1 for CG1 and DG1
	for i in range(0, len(cg1_indices)):
		current = cg1_indices[i]
		constraints.append(inv_learn_avg_copies[current] == 1)

	# Enforce average copy number of 1 for RG1
	for i in range(0, len(rg1_indices)):
		current = rg1_indices[i]
		constraints.append(inv_learn_avg_copies[current] == 1)

	# Enforce monotonic decrease during postG1
	# Inverse of the average curve from 1 to 0.5
	for i in range(1, len(postg1_indices)):
		prev = postg1_indices[i-1]
		current = postg1_indices[i]
		constraints.append(inv_learn_avg_copies[prev] >= inv_learn_avg_copies[current])

		if i == len(postg1_indices)-1:
			constraints.append(inv_learn_avg_copies[current] == 0.5)
		elif i == 1:
			constraints.append(inv_learn_avg_copies[prev] == 1.0)

	# Halted cells, copy number of 1
	constraints.append(inv_learn_avg_copies[m-1] == 1.0)

	prob = cp.Problem(objective, constraints)
	result = prob.solve(solver=solver, warm_start=True, verbose=False, eps=1e-5)

	return result, f, inv_learn_avg_copies.value, predicted_g.value


class RealDataReplicationDeconvolution():
	"""Class to load chromosome data and deconvolve.

	todo: decide if this class will deconvolve all chromosomes/determine
	the average copy number curve for all chromosomes...
	"""
	def __init__(self, chr=10):
		
		from src.mnase_replication_timing_analysis import MNaseOriginAnalysis

		mnase_analysis_rep1 = MNaseOriginAnalysis()
		mnase_analysis_rep1.load_mnase_data(replicate=1, chromosome=chr)

		mnase_analysis_rep1.compute_sliding_window_counts_all_times()
		unnormalized_total_occupancy = mnase_analysis_rep1.all_counts_unnormalized_df
		normalized_total_occ = unnormalized_total_occupancy / \
			unnormalized_total_occupancy.mean(axis=0).values.reshape((1, -1))

		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.normalized_occupancy = normalized_total_occ

