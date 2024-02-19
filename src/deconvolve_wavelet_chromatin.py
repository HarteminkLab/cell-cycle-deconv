
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.wavelets_2d_linalg import wave2d_decomposition, create_wavelet2d_convolution_matrices, wave2d_reconstruction


def deconvolve_wavelet_chromatin(model, H, g, image_shape,
		solver=cvxpy.MOSEK, verbose=False):
	
	# --------------- Wavelet definitions -------------

	wavelet = pywt.Wavelet('bior2.2')

	dec_lo, dec_hi = wavelet.dec_lo, wavelet.dec_hi
	rec_lo, rec_hi = wavelet.rec_lo, wavelet.rec_hi

	(decomp_mats, recon_mats) = create_wavelet2d_convolution_matrices(wavelet, image_shape)

	# --------------------------------------------------


	# We will add a very small value to g, to avoid divide by zero errors
	eps = 1e-5
	g = g + eps

	gamma = model.gamma
	factor_fb = 1.5

	from src.helpers import get_wavelet_kernel

	f_it = model.get_f_it()	
	f_b = model.get_f_b()

	# Mirroring
	f_b_mirror = np.concatenate((f_b, f_b))
	f_it_mirror = np.concatenate((f_it, np.flip(f_it)))
	factor_fb = 1.5

	# Add the wavelet smoothing constraint to the convex optimization
	W1 = get_wavelet_kernel(len(f_it_mirror))
	W2 = get_wavelet_kernel(len(f_b))
	W2pad = np.zeros((len(f_b), len(f_b)))
	W2 = np.concatenate((np.concatenate((W2, W2pad)), 
						 np.concatenate((W2pad, np.fliplr(W2)))), axis=1)

	g_mean = g.mean()


	# -------- Define the optimization ------------

	# f whose rows span the columns of H
	# and columns are the length of g's columns
	f = cvxpy.Variable((H.shape[1], g.shape[1]))

	# The fitting constraint of HF / g
	elementwise_result = cvxpy.multiply(H@f, 1.0/g) - 1

	# The smoothing constraints
	smooth_f_it_result = W1@f[f_it_mirror]
	smooth_f_b_result = W2@f[f_b_mirror]

	n = H.shape[0]
	u = H.shape[1]
	m = g.shape[1]

	# H (n x u)
	# f (u x m)
	# g (n x m)

	elementwise_result = cvxpy.multiply(H@f, 1.0/g) - 1
	
	# Can we apply an L1 norm on the coefficients, set them to 0
	# if they get close to zero?
	l1_norm_on_coeffs = 0

	# For each of the f images, when we decompose and represent these
	# images in wavelet coefficient space, can we drop off coefficients
	# such that the image is still effectively represented with fewer
	# wavelets?
	for i in range(u):

		# Get the f image row we are going to reconstruct
		f_img = f[i, :].reshape(image_shape)

		# Decompose the image into coefficients
		# We can place L1 norms on these somehow.
		(LL, LH, HL, HH) = wave2d_decomposition(f_img, decomp_mats)

		# Try setting an L1 norm on the coefficients, to drop them off to zero if we can
		l1_norm_on_coeffs += cvxpy.sum(cvxpy.abs(LL) + cvxpy.abs(HL) + cvxpy.abs(LH) + cvxpy.abs(HH))


	objective = cvxpy.Minimize(

		# Compute the sum of squares on the result
		cvxpy.sum_squares(elementwise_result)

		# Like-wise, for smoothing compute the l1 norm along each column and compute the sum
		+ gamma * (cvxpy.sum(cvxpy.abs(smooth_f_it_result)) 
		+ factor_fb * cvxpy.sum(cvxpy.abs(smooth_f_b_result)))/g_mean  

		+ (0.001)*l1_norm_on_coeffs
	)

	constraints = [f >= 0]

	# -------- End definition of the optimization ------------

	# Perform the convex optimization
	prob = cvxpy.Problem(objective, constraints)

	# The epsilon value affects the precision of the solver
	result = prob.solve(solver=solver, warm_start=True, verbose=verbose, eps=1e-4)

	# ------- Compute the smoothing norm and fitting/residual norms --------------

	f = f.value

	# We will use the non-mirrored wavelet kernel sizes, because we are operating on the 
	# final f values
	W1 = get_wavelet_kernel(len(f_it))
	W2 = get_wavelet_kernel(len(f_b))

	# Extending the deconvolution a matrix form, 
	# The norm is computing us the Frobeius norm
	# Which is equivalent to the sum of squares of the
	# individual elements in the matrix result
	# Normalize by the result by the size of the grid, m
	matmul_res = np.matmul(H, f) / g - 1
	rn = np.sum(matmul_res**2) / m # Equivalent to: np.linalg.norm(matmul_res, ord='fro')**2 / m

	# Extending the smoothing term, is a little trickier
	# There is no predefined name for the L1 norm type of
	# computation on a matrix, so we manually take the absolute values
	# and take the sum.
	# Normalize by the gene expression level and the size of the grid, m
	f_it_matmul_res = np.matmul(W1, f[f_it])
	f_b_matmul_res = np.matmul(W2, f[f_b])
	sn = (np.sum(np.abs(f_it_matmul_res)) + 
		factor_fb * (np.sum(np.abs(f_b_matmul_res)))) / g_mean / m

	return f, rn, sn