
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

		# Convex optimization
		n, m = self.H.shape
		f_non_padded_indices = np.arange(m)

		if self.padding_type == 'both':

			# Designate the where the padding indices will align to

			# For initial branch, 64 length
			#
			#  RG1 padding | initial branch  64 | postG1 padding 
			# 

			# For the tb branches
			#	
			#  CG1 padding   64  |  top branch   |   bottom branch  |   post G1  64 padding
			#

			from src.helpers import compute_closest_pow2

			# The size of the padded subsets will be equal for simplicity
			size_of_tb_padded = len(f_tb)*2
			size_of_i_padded = size_of_tb_padded

			# Padding for each of the smoothing criteria
			tb_padding = size_of_tb_padded - len(f_tb)
			i_padding =  size_of_i_padded - len(f_i)

			i_padding_2 = i_padding//2
			tb_padding_2 = tb_padding//2

			padding = i_padding + tb_padding

			f_padded = cp.Variable(m+padding)
			f_i_padding_indices = np.arange(m, m+i_padding)
			f_tb_padding_indices = np.arange(m+i_padding, m+i_padding+tb_padding)

			f_i_padded = np.concatenate([f_i_padding_indices[:i_padding_2], 
			 	f_i, f_i_padding_indices[i_padding_2:]])
			f_tb_padded = np.concatenate([f_tb_padding_indices[:tb_padding_2], 
			 	f_tb, f_tb_padding_indices[tb_padding_2:]])

		elif self.padding_type == 'none':

			padding = 0
			padding_2 = 0

			f_padding_indices = np.arange(m, m+padding)
			f_padded = cp.Variable(m+padding)

			f_i_padded = f_i
			f_tb_padded = f_tb

		else:
			raise ValueError(f"Unknown padding type: {self.padding_type}")

		W_i = get_wavelet_kernel(len(f_i_padded))
		W_tb = get_wavelet_kernel(len(f_tb_padded))

		f_non_replicative = f_padded[f_non_padded_indices]
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
		coeffs_weights_W_i = get_level_based_weights(len(f_i_padded))
		coeffs_weights_W_tb = get_level_based_weights(len(f_tb_padded))

		# Let's use the longer coefficients weights, to enforce smoothing that is consistent between
		# the two different subsets
		smooth_f_i_result = cp.multiply(W_i@f_padded[f_i_padded], coeffs_weights_W_i)
		smooth_f_tb_result = cp.multiply(W_tb@f_padded[f_tb_padded], coeffs_weights_W_tb)

		objective = cp.Minimize(

			# L2 fitting norm
			cp.square(cp.norm(elementwise_result, 2)) + 

			# top and bottom branches are 10% longer than initial
			# initial s doubled, because top and bottom are concatenated for
			# postg1-c/dg1 smoothing
			+ self.gamma * cp.sum(cp.abs(smooth_f_i_result))*2.2
			+ self.gamma * cp.sum(cp.abs(smooth_f_tb_result))
		)

		# Constraint for halted cells
		constraints = [f_padded >= 0, f_padded[f_i[0]] == f_padded[f_t[-1]+1]]

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=cp.MOSEK)

		# Convert it into a numpy array
		f = f_padded[f_non_padded_indices].value

		self.f_non_padded_indices = f_non_padded_indices
		self.f = f
		self.f_padded = f_padded.value
		self.f_i_padded = f_i_padded
		self.f_tb_padded = f_tb_padded

		self.W_i = W_i
		self.W_tb = W_tb

		sn = cp.mean(cp.abs(smooth_f_i_result)).value + cp.mean(cp.abs(smooth_f_tb_result)).value
		rn = cp.square(cp.pos(cp.norm(elementwise_result))).value

		self.rn = rn
		self.sn = sn
		self.f = f
