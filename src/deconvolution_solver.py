
import numpy as np
import cvxpy as cp
from src.helpers import get_wavelet_kernel

class DeconvolutionSolver(object):

	def __init__(self, config, g, H, gamma, N=None, f_replication=None,
		b=None, padding_type='both', obj_error_mode='additive', kappa=5e-3):

		n, m = H.shape

		if N is None:
			N = np.eye(n)

		if f_replication is None:
			f_replication = np.ones(m)

		if b is None:
			b = 1

		self.obj_error_mode = obj_error_mode
		self.config = config
		self.g = g
		self.H = H
		self.N = N
		self.f_replication = f_replication
		self.b = b
		self.kappa = kappa

		self.gamma = gamma
		self.padding_type = padding_type
		
	def deconvolve(self):

		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')
		f_b = self.config.get_Hpositions_for_branch('b')
		f_cg1 = self.config.get_Hpositions_for_phase('CG1')
		f_rg1 = self.config.get_Hpositions_for_phase('RG1')
		f_dg1 = self.config.get_Hpositions_for_phase('DG1')
		f_postg1 = self.config.get_Hpositions_for_phase('postG1')

		# Smoothing will be enforced by each branch separately,
		# and by enforcing smoothing going into each of the mother/daughter branches
		f_recovery_smoothing_indices = np.concatenate([f_rg1, f_postg1])
		f_top_smoothing_indices = np.concatenate([f_postg1, f_cg1, f_postg1])
		f_bottom_smoothing_indices = np.concatenate([f_postg1, f_dg1, f_postg1])

		from src.helpers import compute_closest_pow2

		# Convex optimization
		n, m = self.H.shape
		f_non_padded_indices = np.arange(m)

		if self.padding_type == 'both':

			# Designate the where the padding indices will align to

			# For the t branch, smooth transition of cg1 going into postG1
			# Pad the ends of post G1 (reuse for DG1 model
			#
			#	
			#   postG1 | CG1 | postG1
			#
			#   42 + 22 + 42 = 106
			#
			#   padding of 22  (11 on each side)

			# Likewise for bottom branch smoothing
			#	
			#   postG1 | DG1 | postG1
			#
			# Note: the possibility of smoothing going into DG1 and CG1 from the padded edges

			# For initial branch, the left end does not require smoothing from
			# a previous phase
			#
			#    RG1 | postG1
			# 
			#            22 + 42 = 64
			#
			# So pad 64, (32 on each side), the right end can be reused and is longer than the previous 
			# two models
			#

			from src.helpers import compute_closest_pow2

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

			# Now we will designate how much to extend the f vector and where to place the new padded indices
			# We will need 32 for the right side of postG1 for initial 
			# (11 of which can be repeated for the top and bottom padding of postG1)
			# 
			# We need 11 padding for the start of postG1 for the top and bottom branches
			#
			# And 32 for the recovery G1 left side
			#
			#   Left:  
			#            11 (start of postG1 for top and bottom)
			#            32 (start of RG1)
			#   
			#   Right: 
			#            32 (postG1 for RG1, 11 of which is reused for top and bottom)
			#
			#   75 total additional padding

			# Thus we can calculate how much padding we will need for the final padded f vector
			number_of_unique_padding = initial_padding + tb_padding_2

			f_padded_variation = cp.Variable(m+number_of_unique_padding)

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

		elif self.padding_type == 'none':

			padding = 0
			padding_2 = 0

			f_padding_indices = np.arange(m, m+padding)
			f_padded_variation = cp.Variable(m+padding)

			f_i_padded = f_i
			f_tb_padded = f_tb

		else:
			raise ValueError(f"Unknown padding type: {self.padding_type}")

		# All of the wavelets should be the same size
		W_itb = get_wavelet_kernel(len(f_recovery_padded_indices))

		# Model a baseline value, so smoothing constraints are applied to
		# variations on the baseline
		f_baseline = cp.Variable(1)
		f_non_replicative = (f_padded_variation[f_non_padded_indices]+f_baseline)
		self.f_baseline = f_baseline

		f_replication = self.f_replication

		f_combined = cp.multiply(f_non_replicative, f_replication)

		H = self.H
		g = self.g
		N = self.N
		b = self.b

		eps = 1e-5

		if self.obj_error_mode == 'multiplicative':
			elementwise_result = (N@H@f_combined*b)/(g+eps) - 1
		elif self.obj_error_mode == 'additive':
			elementwise_result = (N@H@f_combined*b+eps) - (g+eps)
		else:
			raise ValueError(f"Unimplemented objective error mode: {self.obj_error_mode}")

		from src.helpers import get_level_based_weights

		# Add weight to higher frequency coefficients, to discourage jaggedness
		coefficient_weights_itb = get_level_based_weights(len(f_recovery_padded_indices))

		# Smooth against variations of the baseline
		smooth_f_i_result = cp.multiply(W_itb@(f_padded_variation[f_recovery_padded_indices]-f_baseline), coefficient_weights_itb)
		smooth_f_t_result = cp.multiply(W_itb@(f_padded_variation[f_top_padded_indices]-f_baseline), coefficient_weights_itb)
		smooth_f_b_result = cp.multiply(W_itb@(f_padded_variation[f_bottom_padded_indices]-f_baseline), coefficient_weights_itb)

		fit_norm_result = cp.square(cp.norm(elementwise_result, 2))

		# Smooth each branch separately and weigh by the proportional length of the
		# branch relative to the bottom branch (the longest)

		# Testing adjustments to smoothing weights of the three branches
		# They appear to have minimal effect on gene expression.
		smooth_result = (cp.sum(cp.abs(smooth_f_i_result)) * 1 +
						 cp.sum(cp.abs(smooth_f_t_result)) * 1 +
						 cp.sum(cp.abs(smooth_f_b_result)) * 1)

		# todo: These weights may be too low for the bottom branch, and introduces the
		#       DG1 bias which depicts a greater amount of variability.

		kappa = self.kappa

		tb_regularization_result = f_padded_variation[f_dg1] - f_padded_variation[f_cg1]

		# L2 norm
		# cg1_dg1_regularization_result = cp.square(cp.norm(tb_regularization_result, 2))

		# L1 norm
		cg1_dg1_regularization_result = cp.sum(cp.abs(tb_regularization_result))

		objective = cp.Minimize(

			# L2 fitting norm
			fit_norm_result + 
			self.gamma * smooth_result +
			self.kappa * cg1_dg1_regularization_result
		)

		# Constraint for halted cells, non-negativity, and upper bounds to improve speed
		constraints = [f_padded_variation >= 0, f_baseline >= 0, # non-negativity
			f_padded_variation[f_i[0]] == f_padded_variation[f_t[-1]+1], # halted cells
		]

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=cp.MOSEK)

		# Convert it into a numpy array
		f = f_padded_variation[f_non_padded_indices].value+f_baseline.value

		self.f_non_padded_indices = f_non_padded_indices
		self.f = f
		self.f_padded_variation = f_padded_variation.value

		self.W_itb = W_itb
		self.coefficient_weights_itb = coefficient_weights_itb
		self.f_recovery_padded_indices = f_recovery_padded_indices
		self.f_top_padded_indices = f_top_padded_indices
		self.f_bottom_padded_indices = f_bottom_padded_indices

		self.tb_regularization_result = cg1_dg1_regularization_result.value
		self.sn = smooth_result.value
		self.rn = fit_norm_result.value
		self.f = f

	def plot_fit(self):

		from matplotlib import pyplot as plt

		config = self.config
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		num_cols = 4

		fig, axs = plt.subplots(1, num_cols, figsize=(16, 3))

		g = self.g

		f = self.f

		max_value = np.concatenate([g, f]).max()
		ylims = -((max_value*0.05)), (max_value*1.05)

		gamma_predicted_g = self.H@f

		ax_row = axs

		ax = ax_row[0]

		ax.plot(g[:], c='black', lw=3, label="Raw data")
		ax.plot(gamma_predicted_g, c='red',
				lw=3, label="Optimal $\\gamma$ solution")
		ax.set_title("Data vs Fit")
		ax.legend()
		ax.set_ylim(*ylims)

		ax = ax_row[1]
		ax.plot(f[i_indices], c='red',
				lw=3)
		ax.set_title("Initial branch")
		ax.set_ylim(*ylims)

		ax = ax_row[2]
		ax.plot(f[b_indices], c='blue',
				lw=3, alpha=0.25)
		ax.plot(f[t_indices], c='red',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Top branch")

		ax = ax_row[3]
		ax.plot(f[t_indices], c='red',
				lw=3, alpha=0.25)
		ax.plot(f[b_indices], c='blue',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Bottom branch")
