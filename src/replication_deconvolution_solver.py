
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
	In this class, we will perform some changes to the chromatin deconvolution 
	to handle the copy number transition from 1 to 2. We expect this transition to be a 
	step-wise transition so we will use Haar. And the smoothing constraints will be 
	different as we expect from postG1 to G1, the contraint no longer needs to be smooth.
	"""

	def __init__(self, config, G, solver=cvxpy.MOSEK, wavelet="Haar"):

		from src.helpers import calcH

		self.config = config
		self.H, Hpos = calcH(config.intervals_wt1, config.WT1_TIMEPOINTS)

		self.solver = solver
		self.wavelet = wavelet
		self.G = G

	def plot_raw_data(self):
		plt.figure(figsize=(4, 2))
		plt.plot(self.G[:, 0])

	def define_deconvolution_problem(self):

		solver = self.solver
		G = self.G
		H = self.H
		
		# We will add a very small value to g, to avoid divide by zero errors
		eps = 1e-5
		G = G + eps

		f_dg1 = self.config.get_Hpositions_for_phase('DG1')
		f_rg1 = self.config.get_Hpositions_for_phase('RG1')
		f_cg1 = self.config.get_Hpositions_for_phase('CG1')
		f_pg1 = self.config.get_Hpositions_for_phase('postG1')

		f_b = self.config.get_Hpositions_for_branch('b')
		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')

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

		# -------------------------------------------------------------------------

		elementwise_result = cvxpy.multiply(H@f, 1.0/G) - 1

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		constraints = [f >= 0]

		# -------------------------------------------------

		objective = cvxpy.Minimize(

			cvxpy.sum(cvxpy.norm(elementwise_result, 'fro')**2)

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
		self.result = self.prob.solve(solver=self.solver, warm_start=True, 
			verbose=self.verbose, eps=1e-4)
		f = self.f.value

		if self.result == float('-inf'):
			raise ValueError("No result, possibly too low of coverage for this gene")

		# ------- Upon completion, compute the smoothing norm and 
		# fitting/residual norms --------------

		H = self.H
		G = self.G
		g_mean = G.mean()

		eps = 1e-5
		G = G + eps

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		f_b = self.config.get_Hpositions_for_branch('b')
		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')

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
		rn = np.sum(matmul_res**2) / m 
		# Equivalent to: np.linalg.norm(matmul_res, ord='fro')**2 / m

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
		self.f = f

		return f, rn, sn, l1_norm_on_coeffs

	def plot_result(self):
		plt.figure(figsize=(13, 2))
		config = self.config

		f_dg1 = config.get_Hpositions_for_phase('DG1')
		f_rg1 = config.get_Hpositions_for_phase('RG1')
		f_cg1 = config.get_Hpositions_for_phase('CG1')
		f_pg1 = config.get_Hpositions_for_phase('postG1')

		f_rg1_tps = config.get_phase_timepoints_for_phase('RG1')
		f_cg1_tps = config.get_phase_timepoints_for_phase('CG1')
		f_dg1_tps = config.get_phase_timepoints_for_phase('DG1')
		f_pg1_tps = config.get_phase_timepoints_for_phase('postG1')

		plt.subplot(1, 4, 1)
		plt.plot(f_rg1_tps, self.f[f_rg1])
		plt.plot(f_pg1_tps, self.f[f_pg1])
		plt.title("Recovery")

		plt.subplot(1, 4, 2)
		plt.plot(f_cg1_tps, self.f[f_cg1])
		plt.plot(f_pg1_tps, self.f[f_pg1])
		plt.title("Mother")

		plt.subplot(1, 4, 3)
		plt.plot(f_dg1_tps, self.f[f_dg1])
		plt.plot(f_pg1_tps, self.f[f_pg1])
		plt.title("Daughter")

		plt.subplot(1, 4, 4)
		plt.plot(self.G)
		plt.plot(self.H@self.f)
		plt.title("Predicted/Raw")

def create_mirror(ind_vec):
	ind_vec_n_2 = len(ind_vec) // 2
	ind_vec_mirror = np.concatenate([np.flip(ind_vec[:ind_vec_n_2]), ind_vec, 
		np.flip(ind_vec[-ind_vec_n_2:])])
	return ind_vec_mirror
