
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel
from src.utils import print_fl


from src.wavelets_2d_linalg import decompose_flattened_kron_coeffs, \
	create_kron_wavelet2d_convolution_matrices


class ChromatinDeconvolveSolver:
	"""Class to handle chromatin deconvolution, will be useful for scanning 
	for gamma values and reusing the same
	problem definition"""

	def __init__(self, config, G, N, b, f_replication,
	 	solver=cvxpy.MOSEK, wavelet="Symmlet", padding_type='both', subsample=-1):

		self.config = config
		self.H = config.H
		self.solver = solver
		self.wavelet = wavelet
		self.G = G
		self.N = N
		self.b = b
		self.f_replication = f_replication
		self.padding_type = padding_type

		self.original_G = G.copy()

		if subsample > 0:

			k = subsample

			print_fl(f"Subsampling to top {k} sites")
			G_max_df = pd.DataFrame({'max_value': G.max(axis=0), 'index': np.arange(G.shape[1])})
			highest_Gs = G_max_df.sort_values('max_value', ascending=False).head(k)

			highest_indices = G.argmax(axis=1)
			print("Indices with the highest max value: ", highest_indices)
			selected_indices = highest_Gs.index.values
			selected_examples_G = G[:, selected_indices]
			self.G = selected_examples_G

	def compute_predicted_G(self, F=None):
		N, H, b, f_replication = self.N, self.H, self.b, self.f_replication

		if F is None: F = self.F

		predicted_G = N@H@np.multiply(F, f_replication[:, None])*b
		return predicted_G


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
		self.gamma = gamma

		eps_cutoff = 1e-3

		for i in range(m):

			# Set the solver's G value
			current_g = self.G[:, i]

			if current_g.max() < eps_cutoff:
				# Skip
				pass

			else:

				from src.deconvolution_solver import DeconvolutionSolver

				deconvolution_solver = DeconvolutionSolver(self.config, current_g, 
					self.H, gamma=gamma, padding_type=self.padding_type,
					N=self.N, f_replication=self.f_replication, b=self.b)

				try:
					deconvolution_solver.deconvolve()
					current_f = deconvolution_solver.f
					current_sn = deconvolution_solver.sn
					current_rn = deconvolution_solver.rn
				except cvxpy.error.SolverError:
					continue

				deconvolved_f_value[:, i] = current_f

				# Keep a running rn and sn, divide by m
				# such that the final values will be the mean
				running_rn += current_rn / m
				running_sn += current_sn / m

			if verbose_progress and i % 500 == 0:
				timer.print_time(f"{i}/{m}")

		self.deconvolved_f_value = deconvolved_f_value
		self.F = deconvolved_f_value
		self.rn = running_rn
		self.sn = running_sn
		self.deconvolution_solver = deconvolution_solver
		self.predicted_G = self.compute_predicted_G()

		if verbose:
			timer.print_time(f"Completed")

		return self.deconvolved_f_value


def example_N_frep_b(config):

	n, m = config.H.shape

	padding_type = 'both'

	b = 1

	chr4_n = np.array([1.00442364, 1.00347336, 0.97773201, 0.8061213 , 0.66388052,
	       0.84322967, 0.99955218, 1.00893152, 0.97884289, 0.85742082,
	       0.75599297, 0.76982516, 0.84182678, 0.97059662, 0.99078551,
	       0.91119956])

	N = np.diag(chr4_n)

	# f_rep = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
	#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
	#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
	#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
	#        1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
	#        2., 1.])
	f_rep = np.ones(m)

	return N, f_rep, b
