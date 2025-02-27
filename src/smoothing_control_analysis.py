import numpy as np
from src.helpers import get_wavelet_kernel
from src.helpers import compute_closest_pow2
import cvxpy as cp


def test_fit_f_g(g, config, gamma, kappa):

	f_i = config.get_Hpositions_for_branch('i')
	f_t = config.get_Hpositions_for_branch('t')
	f_b = config.get_Hpositions_for_branch('b')
	f_cg1 = config.get_Hpositions_for_phase('CG1')
	f_rg1 = config.get_Hpositions_for_phase('RG1')
	f_dg1 = config.get_Hpositions_for_phase('DG1')
	f_postg1 = config.get_Hpositions_for_phase('postG1')

	# Smoothing will be enforced by each branch separately,
	# and by enforcing smoothing going into each of the mother/daughter branches
	f_recovery_smoothing_indices = np.concatenate([f_rg1, f_postg1])
	f_top_smoothing_indices = np.concatenate([f_postg1, f_cg1, f_postg1])
	f_bottom_smoothing_indices = np.concatenate([f_postg1, f_dg1, f_postg1])

	# Convex optimization
	n, m = config.H.shape
	f_non_padded_indices = np.arange(m)

	# Top padding is defined by the amount of padding needed to reach
	# a power of 2 (128), pad 11 on each side
	# Bottom padding will be identical to top
	tb_total_padded_len = compute_closest_pow2(len(f_top_smoothing_indices))
	tb_padding = tb_total_padded_len-len(f_top_smoothing_indices)
	tb_padding_2 = tb_padding//2

	# Recovery padding is defined
	# the recovery/initial branch will be a power of 2 (64) so add 1 to get the next highest
	# power of two, should also be 128 (equal to top and bottom's total padded indices)
	initial_total_padded_len = compute_closest_pow2(len(f_recovery_smoothing_indices)+1)
	initial_padding = initial_total_padded_len-len(f_recovery_smoothing_indices)
	initial_padding_2 = initial_padding//2

	# Thus we can calculate how much padding we will need for the final padded f vector
	number_of_unique_padding = initial_padding + tb_padding_2

	# NEW: Add a baseline variable for the mean component
	f_baseline = cp.Variable(1)
	
	# The f_padded now represents deviations from the baseline
	f_padded = cp.Variable(m+number_of_unique_padding)

	# Now let's designate which indices belong to which of the padding assignments from above
	f_padding_indices = np.arange(m, m+number_of_unique_padding)

	# Assign the initial branch paddings first
	left_initial_padding_indices = np.arange(0, initial_padding_2) # 0-32
	right_initial_padding_indices = np.arange(initial_padding_2, initial_padding) # 32-64

	# Top and bottom branch paddings (will be duplicated)
	# Left side is unique and designated for the start of postG1
	left_tb_padding_indices = np.arange(initial_padding, initial_padding+tb_padding_2) # 64-75

	# Right side will be reused from the initial right-padding
	# 11 of the first indices of the initial right padding (postG1's right side)
	right_tb_padding_indices = right_initial_padding_indices[0:tb_padding_2] # 32-43

	# Define the padded indices that will be used for the smoothing
	f_recovery_padded_indices = np.concatenate([
		f_padding_indices[left_initial_padding_indices], 
		f_recovery_smoothing_indices,
		f_padding_indices[right_initial_padding_indices]])

	f_top_padded_indices = np.concatenate([
		f_padding_indices[left_tb_padding_indices],
		f_top_smoothing_indices,
		f_padding_indices[right_tb_padding_indices]])

	f_bottom_padded_indices = np.concatenate([
		f_padding_indices[left_tb_padding_indices],
		f_bottom_smoothing_indices,
		f_padding_indices[right_tb_padding_indices]])

	# All of the wavelets should be the same size
	W_itb = get_wavelet_kernel(len(f_recovery_padded_indices))

	f_non_replicative = f_padded[f_non_padded_indices]

	# NEW: Add baseline to the variations for the final solution
	f_combined = f_baseline + f_non_replicative

	from src.helpers import get_level_based_weights

	elementwise_result = (config.H@f_combined) - (g)

	# Add weight to higher frequency coefficients, to discourage jaggedness
	coefficient_weights_itb = get_level_based_weights(len(f_recovery_padded_indices))
	
	# Let's use the longer coefficients weights, to enforce smoothing that is consistent between
	# the two different subsets
	smooth_f_i_result = cp.multiply(W_itb@(f_padded[f_recovery_padded_indices] - f_baseline), 
	                                coefficient_weights_itb)
	smooth_f_t_result = cp.multiply(W_itb@(f_padded[f_top_padded_indices] - f_baseline), 
	                                coefficient_weights_itb)
	smooth_f_b_result = cp.multiply(W_itb@(f_padded[f_bottom_padded_indices] - f_baseline), 
	                                coefficient_weights_itb)

	fit_norm_result = cp.square(cp.norm(elementwise_result, 2))

	smooth_result = (cp.sum(cp.abs(smooth_f_i_result)) * 0.9 +
	                 cp.sum(cp.abs(smooth_f_t_result)) * 1 +
	                 cp.sum(cp.abs(smooth_f_b_result)) * 1.4)

	tb_regularization_result = f_padded[f_dg1] - f_padded[f_cg1]
	cg1_dg1_regularization_result = cp.square(cp.norm(tb_regularization_result, 2))

	objective = cp.Minimize(
	    fit_norm_result + 
	    gamma * smooth_result +
	    kappa * cg1_dg1_regularization_result
	)

	# Constraint for halted cells, add constraint that f_padded should be non-negative
	constraints = [f_combined >= 0, f_combined[f_i[0]] == f_combined[f_t[-1]+1]]

	prob = cp.Problem(objective, constraints)
	result = prob.solve(solver=cp.MOSEK)

	return fit_norm_result, smooth_result, f_padded, f_combined