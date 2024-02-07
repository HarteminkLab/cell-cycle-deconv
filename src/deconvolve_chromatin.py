
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

	W1 = get_wavelet_kernel(len(f_it_mirror))
	W2 = get_wavelet_kernel(len(f_b))
	W2pad = np.zeros((len(f_b), len(f_b)))
	W2 = np.concatenate((np.concatenate((W2, W2pad)), 
						 np.concatenate((W2pad, np.fliplr(W2)))), axis=1)

	# Add the wavelet smoothing constraint to the convex optimization
	objective = cvxpy.Minimize(cvxpy.square(cvxpy.pos(cvxpy.norm(H@f/g - 1))) 
		+ gamma * (cvxpy.norm(W1@f[f_it_mirror], 1) 
		+ factor_fb * cvxpy.norm(W2@f[f_b_mirror], 1))/g.mean())

	# Where f is non-negative
	if allow_negative:
		constraints = []
	else:
		constraints = [f >= 0]

	# Perform the convex optimization
	prob = cvxpy.Problem(objective, constraints)
	result = prob.solve(solver=cvxpy.CLARABEL)

	# ------- Compute the smoothing norm and fitting/residual norms --------------

	# We will use the non-mirrored wavelet kernel sizes, because we are operating on the 
	# final f values
	W1 = get_wavelet_kernel(len(f_it))
	W2 = get_wavelet_kernel(len(f_b))

	sn = (np.linalg.norm(np.matmul(W1, f.value[f_it]), 1) +
		np.linalg.norm(np.matmul(W2, f.value[f_b]), 1)) / np.mean(g)
	rn = np.square(np.clip(np.linalg.norm(np.matmul(model.H, f.value) / g - 1), 0, None))

	return f.value, rn, sn
