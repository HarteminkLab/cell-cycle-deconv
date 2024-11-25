
import numpy as np
import cvxpy as cp
from src.helpers import get_wavelet_kernel

class DeconvolutionSolver(object):

	def __init__(self, config, g, H, gamma, padding_type='both'):
		self.config = config
		self.g = g
		self.H = H
		self.gamma = gamma
		self.padding_type = padding_type
		
	def deconvolve(self):

		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')

		f_it = np.concatenate([f_i, f_t])

		W_it = get_wavelet_kernel(len(f_it))

		# Convex optimization
		n, m = self.H.shape

		padding = len(f_it)

		f_fit_indices = np.arange(m)
		f_padding_indices = np.arange(m, m+padding)
		f_padded = cp.Variable(m+padding)

		# Pad the start, this allows for smoother transitions for the
		# MNase-seq data in aggregate
		if self.padding_type == 'left':
			f_it_padded = np.concatenate([f_padding_indices[:padding], f_it])

		# Pad both ends, this works well for the gene expression
		elif self.padding_type == 'both':
			f_it_padded = np.concatenate([f_padding_indices[:padding//2], 
			 	f_it, f_padding_indices[padding//2:]])

		elif self.padding_type == 'none':

			padding = 0
			f_fit_indices = np.arange(m)
			f_padding_indices = np.arange(m, m+padding)
			f_padded = cp.Variable(m+padding)
			f_it_padded = np.concatenate([f_padding_indices[:padding//2], 
			 	f_it, f_padding_indices[padding//2:]])

		else:
			raise ValueError(f"Unknown padding type: {self.padding_type}")

		W_it = get_wavelet_kernel(len(f_it_padded))

		# ---------------------------------------------

		# There are twice as many t and b indices compared to i
		# Factor based on time in recovery compared to t and b
		# so multiply i's smoothing term by 2

		# Padding smoothing result
		smooth_f_it_result = W_it@f_padded[f_it_padded]

		objective = cp.Minimize(

			# Fitting norm
			cp.square(cp.pos(cp.norm(self.H@f_padded[f_fit_indices]/self.g - 1))) + 

			# Smoothing norm for it
			+ self.gamma * cp.sum(cp.abs(smooth_f_it_result))/self.g.mean()  
		)

		constraints = [f_padded >= 0]

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=cp.MOSEK)

		# Convert it into a numpy array
		f = f_padded[f_fit_indices].value
		self.f_padded = f_padded.value

		sn = (np.linalg.norm(np.matmul(W_it, f_padded[f_it_padded].value), 1)) / np.mean(self.g)
		rn = np.square(np.clip(np.linalg.norm(np.matmul(self.H, f) / (self.g) - 1), 0, None))

		return f, sn, rn
