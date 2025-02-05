
import numpy as np
import cvxpy as cp
from src.helpers import get_wavelet_kernel

class DeconvolutionSolver(object):

	def __init__(self, config, g, H, gamma, N=None, f_replication=None,
		b=None, padding_type='both', obj_error_mode='additive'):

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

		self.gamma = gamma
		self.padding_type = padding_type
		
	def deconvolve(self):

		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')
		f_b = self.config.get_Hpositions_for_branch('b')
		f_cg1 = self.config.get_Hpositions_for_phase('CG1')

		f_tb = np.concatenate([f_t, f_b])

		from src.helpers import compute_closest_pow2

		# len_f_it = len(f_it)
		# closest_pow2_it = compute_closest_pow2(len_f_it+len_f_it)

		# Try padding to closest power of 2
		# padding = closest_pow2_it-len_f_it
		padding = 0
		padding_2 = padding//2

		# Convex optimization
		n, m = self.H.shape

		f_fit_indices = np.arange(m)
		f_padding_indices = np.arange(m, m+padding)
		f_padded = cp.Variable(m+padding)

		# Pad the start, this allows for smoother transitions for the
		# MNase-seq data in aggregate
		# if self.padding_type == 'left':
		# 	f_it_padded = np.concatenate([f_padding_indices[:padding], f_it])

		# # Pad both ends, this works well for the gene expression
		# elif self.padding_type == 'both':
		# 	f_it_padded = np.concatenate([f_padding_indices[:padding_2], 
		# 	 	f_it, f_padding_indices[padding_2:]])

		# elif self.padding_type == 'none':

		# 	# No padding
		# 	padding = 0
		# 	padding_2 = 0

		# 	f_fit_indices = np.arange(m)
		# 	f_padding_indices = []
		# 	f_padded = cp.Variable(m)
		# 	f_it_padded = f_it

		# else:
		# 	raise ValueError(f"Unknown padding type: {self.padding_type}")

		# W_it, D_it = get_wavelet_kernel(len(f_it_padded))

		W_i = get_wavelet_kernel(len(f_i))
		W_tb = get_wavelet_kernel(len(f_tb))

		f_non_replicative = f_padded[f_fit_indices]
		f_replication = self.f_replication

		f_combined = cp.multiply(f_non_replicative, f_replication)

		H = self.H
		g = self.g
		N = self.N
		b = self.b

		if self.obj_error_mode == 'multiplicative':
			eps = 1e-5
			elementwise_result = (N@H@f_combined*b)/(g+eps) - 1
		elif self.obj_error_mode == 'additive':
			elementwise_result = (N@H@f_combined*b) - (g)
		else:
			raise ValueError(f"Unimplemented objective error mode: {self.obj_error_mode}")

		def get_level_based_weights(N):
			"""Add a weighting system to the coefficients to discourage higher frequency coefficients.
			We desire smoothing on the earlier more broad coefficient levels"""
			weights = np.ones(N)
			level = 0
			start_idx = 0
			 
			 while start_idx < N:
				# Calculate size of current level
				level_size = 2**level
				end_idx = min(start_idx + level_size, N)
				 
				# Assign weights, higher for later levels (higher frequencies)
				weights[start_idx:end_idx] = 2**(level)  # Can adjust base of exponential
				 
				start_idx = end_idx
				level += 1
				 
			 return weights

		# Attempt to weight higher frequency coefficients as more important to zero out
		coeffs_weights_W_i = get_level_based_weights(len(f_i))
		coeffs_weights_W_tb = get_level_based_weights(len(f_tb))

		smooth_f_i_result = cp.multiply(W_i@f_padded[f_i], coeffs_weights_W_i)
		smooth_f_tb_result = cp.multiply(W_tb@f_padded[f_tb], coeffs_weights_W_tb)

		objective = cp.Minimize(

			# Fitting norm
			cp.square(cp.pos(cp.norm(elementwise_result))) + 

			# top and bottom branches are 10% longer than initial
			# initial s doubled, because top and bottom are concatenated for
			# postg1-c/dg1 smoothing
			+ self.gamma * cp.sum(cp.abs(smooth_f_i_result))*2.2
			+ self.gamma * cp.sum(cp.abs(smooth_f_tb_result))
		)

		# Constraint for halted cells
		constraints = [f_padded >= 0, f_padded[-1] == f_padded[0]]

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=cp.MOSEK)

		# Convert it into a numpy array
		f = f_padded[f_fit_indices].value
		self.f_padded = f_padded.value

		sn = cp.sum(cp.abs(smooth_f_i_result)).value + cp.sum(cp.abs(smooth_f_tb_result)).value
		rn = cp.square(cp.pos(cp.norm(elementwise_result))).value

		return f, sn, rn, W_i
