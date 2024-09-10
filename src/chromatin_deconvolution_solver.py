
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel
from src.utils import print_fl
from src.model import create_mirror


from src.wavelets_2d_linalg import decompose_flattened_kron_coeffs, \
	create_kron_wavelet2d_convolution_matrices


class ChromatinDeconvolveSolver:
	"""Class to handle chromatin deconvolution, will be useful for scanning 
	for gamma values and reusing the same
	problem definition"""

	def __init__(self, config, H, G, solver=cvxpy.MOSEK, wavelet="Symmlet"):

		self.config = config
		self.solver = solver
		self.wavelet = wavelet
		self.H = H
		self.G = G

		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')
		f_it = np.concatenate([f_i, f_t])
		f_it = create_mirror(f_it)

		self.W = get_wavelet_kernel(len(f_it), type=self.wavelet)

		self.f_it = f_it


	def deconvolve_G_iteratively(self, gamma, verbose=False, verbose_progress=True):
		"""Iteratively deconvolve columns of G, appears to be more accurate
		as the optimization can strictly treat each problem independently"""

		from src.timer import Timer
		timer = Timer()

		deconvolved_f_value = np.zeros((self.H.shape[1], self.G.shape[1]))

		# Setup of the solver with single dimension G
		m = self.G.shape[1]
		running_sn = 0
		running_rn = 0

		for i in range(m):

			# Set the solver's G value
			current_G = self.G[:, i:i+1]
			self.define_deconvolution_problem(current_G)

			try:
				self.solve(gamma_value=gamma, verbose=verbose)
			except cvxpy.error.SolverError:
				print(f"Error solving i={i}, gamma={gamma}. Skipping.")
				continue

			current_f = self.f.value.flatten()
			deconvolved_f_value[:, i] = current_f

			running_rn += self.rn / m
			running_sn += self.sn / m

			if verbose_progress and i % 500 == 0:
				timer.print_time(f"{i}/{m}")

		self.deconvolved_f_value = deconvolved_f_value
		self.rn = running_rn
		self.sn = running_sn
		return self.deconvolved_f_value


	def define_deconvolution_problem(self, G):

		solver = self.solver

		H = self.H
		
		# We will add a very small value to g, to avoid divide by zero errors
		eps = 1e-5
		G = G + eps
		self.current_G = G

		g_mean = G.mean()

		f_it = self.f_it
		W = self.W

		# -------- Define the optimization ------------

		# f whose rows span the columns of H
		# and columns are the length of g's columns
		f = cvxpy.Variable((H.shape[1], G.shape[1]))

		self.gamma = cvxpy.Parameter(nonneg=True, name='gamma')

		# with the updated alpha, the i t and b are approximately all the same length
		# this was previously 2, when i was half the length of the other two branches
		self.factor_i = 1

		# The smoothing constraints
		smooth_f_it_result = W@f[f_it]

		elementwise_result = cvxpy.multiply(H@f, 1.0/G) - 1

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		constraints = [f >= 0]

		# -------------------------------------------------

		objective = cvxpy.Minimize(

			cvxpy.sum(cvxpy.norm(elementwise_result, 'fro')**2)

			# Smoothing along time
			+ self.gamma * cvxpy.sum(cvxpy.abs(smooth_f_it_result))/g_mean  
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
		#         fitting/residual norms --------------

		H = self.H
		G = self.current_G
		g_mean = G.mean()

		eps = 1e-5
		G = G + eps

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

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
		f_it_matmul_res = np.matmul(self.W, f[self.f_it])

		eps = 1e-5
		sn = (self.factor_i * np.sum(np.abs(f_it_matmul_res))) / (g_mean+eps) / m

		l1_norm_on_coeffs = 0
		self.rn, self.sn, self.l1_norm_on_coeffs = rn, sn, l1_norm_on_coeffs

		return f, rn, sn, l1_norm_on_coeffs

