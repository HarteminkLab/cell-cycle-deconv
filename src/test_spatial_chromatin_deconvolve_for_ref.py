
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cvxpy




import pywt
from src.wavelets_2d_linalg import wave2d_decomposition, create_wavelet2d_convolution_matrices, wave2d_reconstruction



def deconvolve_wavelet_chromatin(model, H, g, coeffs_shape, image_shape, allow_negative=False,
		solver=cvxpy.MOSEK, verbose=False):


	# --------------- Wavelet definitions -------------

	# wavelet = pywt.Wavelet('bior2.2')

	# dec_lo, dec_hi = wavelet.dec_lo, wavelet.dec_hi
	# rec_lo, rec_hi = wavelet.rec_lo, wavelet.rec_hi

	# (decomp_mats, recon_mats) = create_wavelet2d_convolution_matrices(wavelet, image_shape)

	# --------------------------------------------------

	# We will add a very small value to g, to avoid divide by zero errors
	eps = 1e-5
	g = g + eps

	from src.helpers import get_wavelet_kernel

	f_it = model.get_f_it()	
	f_b = model.get_f_b()

	g_mean = g.mean()

	# -------- Define the optimization ------------

	# F will now be represented a set of coefficients
	# flattened the coefficients because we can only have a 
	# maximum of 2 dimensions
	# num_coeffs = 4
	# coeffs_shape_len = coeffs_shape[0]*coeffs_shape[1]

	n = H.shape[0] # timepoints
	u = H.shape[1] # cell cycle timepoints
	# m_coeffs = num_coeffs*coeffs_shape_len # number of values to deconvolve
	m_img = image_shape[0]*image_shape[1] # number of values to deconvolve

	# the wavelet coefficients
	f = cvxpy.Variable((u, m_img))

	print(f.shape)
	print(H.shape)
	print(g.shape)

	#f_coeffs = cvxpy.Variable((H.shape[1], m_coeffs))

	# Thus we will need to now calculate f, the image 
	# from our wavelets reconstruction

	#elementwise_result = cvxpy.multiply(H@f, 1.0/g) - 1
	# sum_square_Hf_g = 0
	
	# for i in range(u):

	# 	# f_img = f[i].reshape(image_shape)

	# 	# coeffs = wave2d_decomposition(f_img, decomp_mats)
	# 	# reconstruction = wave2d_reconstruction(coeffs, recon_mats)


	# 	# f_i = reconstruction.reshape((1, -1), order='C')
	# 	H_i = H[:, i].reshape((-1, 1))

	# 	# # Compute the difference between the reconstructed f image
	# 	# cur_i_pred_comparision = cvxpy.multiply(H_i@f_i, 1.0/g) - 1
	# 	# cur_ss = cvxpy.sum_squares(cur_i_pred_comparision)

	# 	#sum_square_Hf_g += cur_ss

	# 	# The non-decomposed f
	# 	curcol = cvxpy.multiply(H_i@f[i].reshape((1, -1), order='C'), 1.0/g) - 1
	# 	cur_ss = cvxpy.sum_squares(curcol)

	# 	sum_square_Hf_g += cur_ss

	# The fitting constraint of HF / g
	elementwise_result = cvxpy.multiply(H@f, 1.0/g) - 1

	m = g.shape[1]

	objective = cvxpy.Minimize(

		# Compute the sum of squares on the result
		cvxpy.sum_squares(elementwise_result)
		# sum_square_Hf_g
	)

	# Where f is non-negative
	if allow_negative:
		constraints = []
	else:
		constraints = [f >= 0]

	# -------- End definition of the optimization ------------

	# Perform the convex optimization
	prob = cvxpy.Problem(objective, constraints)

	# The epsilon value affects the precision of the solver
	result = prob.solve(solver=solver, warm_start=True, verbose=verbose, eps=1e-4)

	# ------- Compute the smoothing norm and fitting/residual norms --------------

	return f#_coeffs
