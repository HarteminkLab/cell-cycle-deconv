
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel, pad_with_subset
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

		f_rg1 = self.config.get_Hpositions_for_phase('RG1')
		f_dg1 = self.config.get_Hpositions_for_phase('DG1')
		f_cg1 = self.config.get_Hpositions_for_phase('CG1')
		f_pg1 = self.config.get_Hpositions_for_phase('postG1')


		# Add the start of S to the G1s
		# f_rg1 = np.concatenate([f_rg1, f_pg1[:5]])
		# f_dg1 = np.concatenate([f_dg1, f_pg1[:5]])
		# f_cg1 = np.concatenate([f_cg1, f_pg1[:5]])


		f_rg1_padded = pad_with_subset(f_rg1)
		f_cg1_padded = pad_with_subset(f_cg1)
		f_dg1_padded = pad_with_subset(f_dg1)

		# Add the end of S to the beginning of G2M
		f_s_sub = f_pg1[:len(f_pg1)//2]
		f_g2m_sub = f_pg1[-len(f_pg1)//2:]

		f_s_padded = pad_with_subset(f_s_sub)
		f_g2m_padded = pad_with_subset(f_g2m_sub)

		# Can we enforce smoothing on S that allows for the step?
		# Perhaps it would need a lower multiplier compared to the other 
		# wavelet enforcements.
		#
		# For now, we can test with half of the post g1 list


		# Appears to work, but we need to enforce the start of S
		# and the end of the G1s
		#
		# and the end of S and the start of G2M


		W1 = get_wavelet_kernel(len(f_rg1_padded), type=self.wavelet)
		W2 = get_wavelet_kernel(len(f_cg1_padded), type=self.wavelet)
		W3 = get_wavelet_kernel(len(f_dg1_padded), type=self.wavelet)
		
		W5 = get_wavelet_kernel(len(f_g2m_padded), type=self.wavelet)


		W4 = get_wavelet_kernel(len(f_s_padded), type="Symmlet")

		g_mean = G.mean()

		# -------- Define the optimization ------------

		# f whose rows span the columns of H
		# and columns are the length of g's columns
		f = cvxpy.Variable((H.shape[1], G.shape[1]))

		self.gamma = cvxpy.Parameter(nonneg=True, name='gamma')

		# i branch is half the length of t and b
		self.factor_i = 1.

		# The smoothing constraints
		smooth_f_i_result = W1@f[f_rg1_padded]
		smooth_f_t_result = W2@f[f_cg1_padded]
		smooth_f_b_result = W3@f[f_dg1_padded]
		smooth_f_s_result = W4@f[f_s_padded]
		smooth_f_g2m_result = W5@f[f_g2m_padded]

		# -------------------------------------------------------------------------

		elementwise_result = cvxpy.multiply(H@f, 1.0/G) - 1

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		constraints = [f >= 0]

		for i in range(1, len(f_s_padded)):
			index = f_s_padded[i]
			prev_index = f_s_padded[i-1]
			constraints.append(f[index] >= f[prev_index])

		constraints.append(f[f_s_padded[0]] == f[f_cg1_padded[0]])
		constraints.append(f[f_s_padded[0]] == f[f_dg1_padded[0]])
		constraints.append(f[f_s_padded[0]] == f[f_rg1_padded[0]])
		constraints.append(f[f_s_padded[-1]] <= f[f_g2m_sub[0]])
		
		for i in range(1, len(f_rg1_padded)):
			index = f_rg1_padded[i]
			prev_index = f_rg1_padded[i-1]
			constraints.append(f[index] == f[prev_index])

		for i in range(1, len(f_dg1_padded)):
			index = f_dg1_padded[i]
			prev_index = f_dg1_padded[i-1]
			constraints.append(f[index] == f[prev_index])

		for i in range(1, len(f_cg1_padded)):
			index = f_cg1_padded[i]
			prev_index = f_cg1_padded[i-1]
			constraints.append(f[index] == f[prev_index])

		for i in range(1, len(f_g2m_sub)):
			index = f_g2m_sub[i]
			prev_index = f_g2m_sub[i-1]
			constraints.append(f[index] == f[prev_index])

		# -------------------------------------------------

		objective = cvxpy.Minimize(

			cvxpy.sum(cvxpy.norm(elementwise_result, 'fro')**2) +

			self.gamma * (cvxpy.sum(cvxpy.abs(smooth_f_i_result)) +
							cvxpy.sum(cvxpy.abs(smooth_f_t_result)) + 
							cvxpy.sum(cvxpy.abs(smooth_f_b_result)) +
							#0.1 * cvxpy.sum(cvxpy.abs(smooth_f_s_result)) +
							cvxpy.sum(cvxpy.abs(smooth_f_g2m_result))
							)/g_mean  
		)

		# -------- End definition of the 	 ------------

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

		self.f = f

		return f

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
