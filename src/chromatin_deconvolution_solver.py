
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

	def __init__(self, config, H, G, N, b, f_replication,
	 	solver=cvxpy.MOSEK, wavelet="Symmlet", padding_type='left'):

		self.config = config
		self.solver = solver
		self.wavelet = wavelet
		self.H = H
		self.G = G
		self.N = N
		self.b = b
		self.f_replication = f_replication
		self.padding_type = padding_type

		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')

		self.f_it = np.concatenate([f_i, f_t])
		self.W_it = get_wavelet_kernel(len(self.f_it))


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

		for i in range(m):

			# Set the solver's G value
			current_g = self.G[:, i]

			from src.deconvolution_solver import DeconvolutionSolver

			deconvolution_solver = DeconvolutionSolver(self.config, current_g, 
				self.H, gamma=gamma, padding_type=self.padding_type,
				N=self.N, f_replication=self.f_replication, b=self.b)

			try:
				current_f, self.sn, self.rn = deconvolution_solver.deconvolve()
			except cvxpy.error.SolverError:
				continue

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
		self.G = G
