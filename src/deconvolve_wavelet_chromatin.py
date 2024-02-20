
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel
from src.utils import print_fl


class ChromatinDeconvolveSolver:
	"""Class to handle chromatin deconvolution, will be useful for scanning for gamma values and reusing the same
	problem definition"""

	def __init__(self, deconv_model, H, G, image_shape, wavelet_name='bior4.4', gamma_prime=0,
				 solver=cvxpy.MOSEK):

		self.deconv_model = deconv_model
		self.solver = solver
		self.wavelet = pywt.Wavelet(wavelet_name)
		self.G = G
		self.H = H
		self.gamma_prime = gamma_prime
		self.image_shape = image_shape


	def define_deconvolution_problem(self):

		solver = self.solver
		wavelet = self.wavelet
		G = self.G
		H = self.H
		gamma_prime = self.gamma_prime
		image_shape = self.image_shape
		
		# We will add a very small value to g, to avoid divide by zero errors
		eps = 1e-5
		G = G + eps

		self.factor_fb = 1.5

		f_it = self.deconv_model.get_f_it()	
		f_b = self.deconv_model.get_f_b()

		# Mirroring
		f_b_mirror = np.concatenate((f_b, f_b))
		f_it_mirror = np.concatenate((f_it, np.flip(f_it)))

		# Add the wavelet smoothing constraint to the convex optimization
		W1 = get_wavelet_kernel(len(f_it_mirror))
		W2 = get_wavelet_kernel(len(f_b))
		W2pad = np.zeros((len(f_b), len(f_b)))
		W2 = np.concatenate((np.concatenate((W2, W2pad)), 
							 np.concatenate((W2pad, np.fliplr(W2)))), axis=1)

		g_mean = G.mean()

		# -------- Define the optimization ------------

		# f whose rows span the columns of H
		# and columns are the length of g's columns
		f = cvxpy.Variable((H.shape[1], G.shape[1]))

		self.gamma = cvxpy.Parameter(nonneg=True, name='gamma')

		# The smoothing constraints
		smooth_f_it_result = W1@f[f_it_mirror]
		smooth_f_b_result = W2@f[f_b_mirror]

		elementwise_result = cvxpy.multiply(H@f, 1.0/G) - 1

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		constraints = [f >= 0]
		
		# ---------- Spatial Wavelet Smoothing -------------

		from src.wavelets_2d_linalg import decompose_flattened_kron_coeffs, \
	    	create_kron_wavelet2d_convolution_matrices

		decomp_kron_mats, reconst_mats = create_kron_wavelet2d_convolution_matrices(wavelet, image_shape)

		if gamma_prime == 0:
			l1_norm_on_coeffs = 0
		else:
			# Decompose the f matrix of flattened images using the kronecker version of the wavelet transformation
			# matrices. Retrieve the wavelet coefficients and compute an L1 norm on these coefficients.
			(LL, HL, LH, HH) = decompose_flattened_kron_coeffs(f, decomp_kron_mats)
			l1_norm_on_coeffs = cvxpy.sum(cvxpy.abs(LL) + cvxpy.abs(HL) + cvxpy.abs(LH) + cvxpy.abs(HH)) / m / u

			# We could try constraining the top end of the coefficient values, if they are getting too high.
			# M = 100
			# constraints += [cvxpy.abs(LL) <= M, cvxpy.abs(HL) <= M, cvxpy.abs(LH) <= M, cvxpy.abs(HH) <= M]

		# -------------------------------------------------

		objective = cvxpy.Minimize(

			# Compute the sum of squares on the result
			cvxpy.sum_squares(elementwise_result)

			# Like-wise, for smoothing compute the l1 norm along each column and compute the sum
			+ self.gamma * (cvxpy.sum(cvxpy.abs(smooth_f_it_result)) 
			+ self.factor_fb * cvxpy.sum(cvxpy.abs(smooth_f_b_result)))/g_mean  

			# How much to apply the l1 norm on the coefficient representation
			+ gamma_prime*l1_norm_on_coeffs
		)

		# -------- End definition of the optimization ------------

		# Perform the convex optimization
		self.prob = cvxpy.Problem(objective, constraints)
		self.f = f

	def solve(self, gamma_value, verbose=False):

		self.gamma.value = gamma_value
		self.verbose = verbose

		# The epsilon value affects the precision of the solver
		result = self.prob.solve(solver=self.solver, warm_start=True, verbose=self.verbose, eps=1e-4)
		f = self.f.value

		# ------- Upon completion, compute the smoothing norm and fitting/residual norms --------------

		H = self.H
		G = self.G
		g_mean = G.mean()

		eps = 1e-5
		G = G + eps

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]
		f_it = self.deconv_model.get_f_it()	
		f_b = self.deconv_model.get_f_b()

		# We will use the non-mirrored wavelet kernel sizes, because we are operating on the 
		# final f values
		W1 = get_wavelet_kernel(len(f_it))
		W2 = get_wavelet_kernel(len(f_b))

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
		f_it_matmul_res = np.matmul(W1, f[f_it])
		f_b_matmul_res = np.matmul(W2, f[f_b])
		sn = (np.sum(np.abs(f_it_matmul_res)) + 
			self.factor_fb * (np.sum(np.abs(f_b_matmul_res)))) / g_mean / m

		if self.gamma_prime > 0:
			# Decompose the f matrix of flattened images using the kronecker version of the wavelet transformation
			# matrices. Retrieve the wavelet coefficients and compute an L1 norm on these coefficients.
			(LL, HL, LH, HH) = decompose_flattened_kron_coeffs(f, decomp_kron_mats)
			l1_norm_on_coeffs = np.sum(np.abs(LL) + np.abs(HL) + np.abs(LH) + np.abs(HH)) / m / u

		else:
			l1_norm_on_coeffs = np.nan

		self.rn, self.sn, self.l1_norm_on_coeffs = rn, sn, l1_norm_on_coeffs

		return f, rn, sn, l1_norm_on_coeffs
