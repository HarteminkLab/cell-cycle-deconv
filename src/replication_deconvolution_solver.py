
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel
from src.utils import print_fl


from src.wavelets_2d_linalg import decompose_flattened_kron_coeffs, \
	create_kron_wavelet2d_convolution_matrices


class ReplicationChromatinDeconvolveSolver:
	"""
	In this class, we will perform some changes to the chromatin deconvolution to handle the copy number transition
	from 1 to 2. We expect this transition to be a step-wise transition so we will use Haar. And the smoothing
	constraints will be different as we expect from postG1 to G1, the contraint no longer needs to be smooth.
	"""

	def __init__(self, deconv_model, H, G, image_shape, solver=cvxpy.MOSEK, wavelet="Haar"):

		self.deconv_model = deconv_model
		self.solver = solver
		self.wavelet = wavelet
		self.G = G
		self.H = H
		self.image_shape = image_shape


	def define_deconvolution_problem(self):

		solver = self.solver
		#spatial_wavelet = self.spatial_wavelet
		G = self.G
		H = self.H
		image_shape = self.image_shape
		
		# We will add a very small value to g, to avoid divide by zero errors
		eps = 1e-5
		G = G + eps

		f_dg1 = self.deconv_model.config.get_Hpositions_for_phase('DG1')
		f_rg1 = self.deconv_model.config.get_Hpositions_for_phase('RG1')
		f_cg1 = self.deconv_model.config.get_Hpositions_for_phase('CG1')
		f_pg1 = self.deconv_model.config.get_Hpositions_for_phase('postG1')

		f_b = self.deconv_model.config.get_Hpositions_for_branch('b')
		f_i = self.deconv_model.config.get_Hpositions_for_branch('i')
		f_t = self.deconv_model.config.get_Hpositions_for_branch('t')

		# The bottom and top branches need to enforce the start
		# of G1 is smooth from the end of postG1, so concatenate those
		# Then mirror the ends to handle edge effects
		f_b_mirror = create_mirror(f_b)
		f_i_mirror = create_mirror(f_i)
		f_t_mirror = create_mirror(f_t)

		W1 = get_wavelet_kernel(len(f_i_mirror), type=self.wavelet)
		W2 = get_wavelet_kernel(len(f_t_mirror), type=self.wavelet)
		W3 = get_wavelet_kernel(len(f_b_mirror), type=self.wavelet)

		g_mean = G.mean()

		# -------- Define the optimization ------------

		# f whose rows span the columns of H
		# and columns are the length of g's columns
		f = cvxpy.Variable((H.shape[1], G.shape[1]))

		self.gamma = cvxpy.Parameter(nonneg=True, name='gamma')

		# i branch is half the length of t and b
		self.factor_i = 2

		# The smoothing constraints
		smooth_f_i_result = W1@f[f_i_mirror]
		smooth_f_t_result = W2@f[f_t_mirror]
		smooth_f_b_result = W3@f[f_b_mirror]
		# f_it = np.concatenate([f_i, f_t])

		g_mean = G.mean()

		# -------- Define the optimization ------------

		# f whose rows span the columns of H
		# and columns are the length of g's columns
		f = cvxpy.Variable((H.shape[1], G.shape[1]))

		self.gamma = cvxpy.Parameter(nonneg=True, name='gamma')

		# i branch is half the length of t and b
		self.factor_i = 2

		# The smoothing constraints
		smooth_f_i_result = W1@f[f_i_mirror]
		smooth_f_t_result = W2@f[f_t_mirror]
		smooth_f_b_result = W3@f[f_b_mirror]

		# -------------------- Smoothing for replication profile ------------------

		# TODO: Testing a prototype idea for replication profile smoothing

		# According to Xin deconvModelBud, we should smooth RG1, CG1, DG1, and postG1 for budding index
		# with the expectation that the G1s are 0 and postG1 is 100
		#
		# For our copy number deconvolution, we will want to deconvolve but allow for S-phase to be
		# freely transition from 1-2 copy number but have RG1, CG1, DG1 all be copy number 1.
		#

		# f_dg1 = self.deconv_model.config.get_Hpositions_for_phase('DG1')
		# f_rg1 = self.deconv_model.config.get_Hpositions_for_phase('RG1')
		# f_cg1 = self.deconv_model.config.get_Hpositions_for_phase('CG1')
		# f_pg1 = self.deconv_model.config.get_Hpositions_for_phase('postG1')

		# print(len(f_dg1))
		# print(len(f_rg1))
		# print(len(f_cg1))
		# print(len(f_pg1))

		# W1 = get_wavelet_kernel(len(f_dg1), type=self.wavelet)
		# W2 = get_wavelet_kernel(len(f_rg1), type=self.wavelet)
		# W3 = get_wavelet_kernel(len(f_cg1), type=self.wavelet)
		# W4 = get_wavelet_kernel(len(f_pg1), type=self.wavelet)

		# # The smoothing constraints
		# smooth_f_i_result = W1@f[f_dg1]
		# smooth_f_t_result = W2@f[f_rg1]
		# smooth_f_b_result = W3@f[f_cg1]
		# smooth_f_p_result = W3@f[f_pg1]

		# -------------------------------------------------------------------------

		elementwise_result = cvxpy.multiply(H@f, 1.0/G) - 1

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		constraints = [f >= 0]

		# -------------------------------------------------

		objective = cvxpy.Minimize(

			# Mathematically these are equivalent:
			#
			# cvxpy.sum(cvxpy.norm(elementwise_result, 2)**2)
			# cvxpy.sum_squares(elementwise_result)
			# 
			# However, there is an implementation detail in cvxpy that favors norm calls over sum of squares:
			#
			# motivated by:
			# https://stackoverflow.com/questions/65526377/cvxpy-returns-infeasible-inaccurate-on-quadratic-programming-optimization-proble
			# https://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms
			# 
			# cvxpy.sum_squares(elementwise_result)
			cvxpy.sum(cvxpy.norm(elementwise_result, 'fro')**2)

			# Smoothing along time
			# + self.gamma * (self.factor_i*cvxpy.sum(cvxpy.abs(smooth_f_i_result)) +
			# 				cvxpy.sum(cvxpy.abs(smooth_f_t_result)) + 
			# 				cvxpy.sum(cvxpy.abs(smooth_f_b_result) +
			# 					      cvxpy.abs(smooth_f_p_result))
			# 				)/g_mean  

			+ self.gamma * (self.factor_i*cvxpy.sum(cvxpy.abs(smooth_f_i_result)) +
							cvxpy.sum(cvxpy.abs(smooth_f_t_result)) + 
							cvxpy.sum(cvxpy.abs(smooth_f_b_result)))/g_mean  
		)

		# -------- End definition of the optimization ------------

		# Perform the convex optimization
		self.prob = cvxpy.Problem(objective, constraints)
		self.f = f

	def solve(self, gamma_value, verbose=False):

		self.gamma.value = gamma_value
		self.verbose = verbose

		# The epsilon value affects the precision of the solver
		self.result = self.prob.solve(solver=self.solver, warm_start=True, verbose=self.verbose, eps=1e-4)
		f = self.f.value

		if self.result == float('-inf'):
			raise ValueError("No result, possibly too low of coverage for this gene")

		# ------- Upon completion, compute the smoothing norm and fitting/residual norms --------------

		H = self.H
		G = self.G
		g_mean = G.mean()

		eps = 1e-5
		G = G + eps

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		f_b = self.deconv_model.config.get_Hpositions_for_branch('b')
		f_i = self.deconv_model.config.get_Hpositions_for_branch('i')
		f_t = self.deconv_model.config.get_Hpositions_for_branch('t')

		# f_it = np.concatenate([f_i, f_t])

		# We will use the non-mirrored wavelet kernel sizes, because we are operating on the 
		# final f values
		W1 = get_wavelet_kernel(len(f_i), type=self.wavelet)
		W2 = get_wavelet_kernel(len(f_t), type=self.wavelet)
		W3 = get_wavelet_kernel(len(f_b), type=self.wavelet)

		# Extending the deconvolution a matrix form, 
		# The norm is computing us the Frobeius norm
		# Which is equivalent to the sum of squares of the
		# individual elements in the matrix result
		# Normalize by the result by the size of the grid, m
		matmul_res = np.matmul(H, f) / G - 1
		rn = np.sum(matmul_res**2) / m # Equivalent to: np.linalg.norm(matmul_res, ord='fro')**2 / m

		# Extending the smoothing term, is a little trickier
		# There is no predefined name for the L1 norm type of
		# computation on a matrix, so we manually take the absolute values
		# and take the sum.
		# Normalize by the gene expression level and the size of the grid, m
		f_i_matmul_res = np.matmul(W1, f[f_i])
		f_t_matmul_res = np.matmul(W2, f[f_t])
		f_b_matmul_res = np.matmul(W3, f[f_b])

		sn = ((self.factor_i * np.sum(np.abs(f_i_matmul_res)) + 
			   				   np.sum(np.abs(f_t_matmul_res)) + 
			   				   np.sum(np.abs(f_b_matmul_res))) / g_mean / m)

		l1_norm_on_coeffs = 0
		self.rn, self.sn, self.l1_norm_on_coeffs = rn, sn, l1_norm_on_coeffs

		return f, rn, sn, l1_norm_on_coeffs

def create_mirror(ind_vec):
	ind_vec_n_2 = len(ind_vec) // 2
	ind_vec_mirror = np.concatenate([np.flip(ind_vec[:ind_vec_n_2]), ind_vec, np.flip(ind_vec[-ind_vec_n_2:])])
	return ind_vec_mirror
