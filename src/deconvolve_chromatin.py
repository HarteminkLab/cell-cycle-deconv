
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cvxpy


def deconvolve_chromatin(model, g, allow_negative=False):
	"""
	Deconvolve the chromatin array


	TODO: this is nearly identical
	to the Model.deconvolve() method.
	So we may want to refactor to just have one method 
	"""

	# We will add a very small value to g, to avoid divide by zero errors
	eps = 1e-5
	g = g + eps

	H = model.H
	gamma = model.gamma
	factor_fb = 1.5

	# f whose rows span the columns of H
	# and columns are the length of g's columns
	f = cvxpy.Variable((H.shape[1], g.shape[1]))

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

	# The 
	elementwise_result = cvxpy.multiply(H@f, 1.0/g) - 1


	# W1 is a (u' x u') matrix
	# and f[f_it] is (u' x m)

	# So, we will need to do the same, l1 norm sum
	# This may not require a change to the sn calculation
	smooth_f_it_result = W1@f[f_it_mirror]
	smooth_f_b_result = W2@f[f_b_mirror]

	m = g.shape[1]

	objective = cvxpy.Minimize(
	    sum(
	    	cvxpy.square(
	    		cvxpy.norm(elementwise_result[:, column], 2)) for column in range(m)
    	)
	 	+ gamma * (sum(cvxpy.norm(smooth_f_it_result[:, column], 1) for column in range(m)) 
	 	+ factor_fb * sum(cvxpy.norm(smooth_f_b_result[:, column], 1) for column in range(m)))/g_mean
	)

	# Where f is non-negative
	if allow_negative:
		constraints = []
	else:
		constraints = [f >= 0]

	# -------- End definition of the optimization ------------

	# Perform the convex optimization
	prob = cvxpy.Problem(objective, constraints)
	result = prob.solve(solver=cvxpy.CLARABEL)

	# ------- Compute the smoothing norm and fitting/residual norms --------------

	f = f.value

	# We will use the non-mirrored wavelet kernel sizes, because we are operating on the 
	# final f values
	W1 = get_wavelet_kernel(len(f_it))
	W2 = get_wavelet_kernel(len(f_b))

	# Take the column-wise l2 norm, then take the mean of these columns
	# this should keep the fitting norm agnostic to the size of the grid
	columnwise_rn_l2 = np.square(np.linalg.norm(np.matmul(H, f) / g - 1, 2, axis=0))
	rn = np.mean(columnwise_rn_l2)

	# Likewise, take the column-wise l1 norms of both smoothing constraints
	# take the mean of the column-wise norms
	sn = (np.linalg.norm(np.matmul(W1, f[f_it]), 1, axis=0).mean() +
      factor_fb * np.linalg.norm(np.matmul(W2, f[f_b]), 1, axis=0).mean()) / g_mean

	return f, rn, sn
