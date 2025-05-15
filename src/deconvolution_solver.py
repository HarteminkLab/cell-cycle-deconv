
import numpy as np
import cvxpy as cp
from src.helpers import get_wavelet_kernel

class DeconvolutionSolver(object):

	def __init__(self, config, g, H, gamma, N=None, f_replication=None,
		b=None, padding_type='both', obj_error_mode='additive', kappa=5e-3,
		data_is_logged=True, unlog_transform=False, log_transform=False):

		n, m = H.shape

		if N is None:
			N = np.eye(n)

		if f_replication is None:
			f_replication = np.ones(m)

		if b is None:
			b = 1

		if data_is_logged and unlog_transform:
			g = 2**g

		if data_is_logged is False and log_transform:
			g = np.log2(g+1)

		self.obj_error_mode = obj_error_mode
		self.config = config
		self.g = g
		self.H = H
		self.N = N
		self.f_replication = f_replication
		self.b = b
		self.kappa = kappa
		self.verbose = False

		self.gamma = gamma
		self.padding_type = padding_type
		
	def deconvolve(self):

		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')
		f_b = self.config.get_Hpositions_for_branch('b')
		f_cg1 = self.config.get_Hpositions_for_phase('CG1')
		f_rg1 = self.config.get_Hpositions_for_phase('RG1')
		f_dg1 = self.config.get_Hpositions_for_phase('DG1')
		f_postg1 = self.config.get_Hpositions_for_phase('postG1')

		# Smoothing will be enforced by each branch separately,
		# and by enforcing smoothing going into each of the mother/daughter branches
		f_i_mirror = np.concatenate([np.flip(f_i), f_i])
		f_t_duplicate = np.concatenate([f_t, f_t])
		f_b_duplicate = np.concatenate([f_b, f_b])

		from src.helpers import compute_closest_pow2

		# Convex optimization
		n, m = self.H.shape

		f_indices = np.arange(m)
		f_variation = cp.Variable(m)

		# Create block matrix structures for wavelets
		W_i_padded = get_wavelet_kernel(len(f_i_mirror), par=5)
		W_t_padded = get_wavelet_kernel(len(f_t_duplicate), par=5)
		W_b_padded = get_wavelet_kernel(len(f_b_duplicate), par=5)

		# Model a baseline value, so smoothing constraints are applied to
		# variations on the baseline
		f_baseline = cp.Variable(1)
		f_non_replicative = f_variation+f_baseline
		self.f_baseline = f_baseline

		f_replication = self.f_replication

		f_combined = cp.multiply(f_non_replicative, f_replication)

		H = self.H
		g = self.g
		N = self.N
		b = self.b

		eps = 1e-5

		if self.obj_error_mode == 'multiplicative':
			elementwise_result = (N@H@f_combined*b)/(g+eps) - 1
		elif self.obj_error_mode == 'additive':
			elementwise_result = N@H@f_combined*b - g
		else:
			raise ValueError(f"Unimplemented objective error mode: {self.obj_error_mode}")

		# Add weight to higher frequency coefficients, to discourage jaggedness
		# Smooth against variations of the baseline
		smooth_f_i_result = W_i_padded@f_variation[f_i_mirror]
		smooth_f_t_result = W_t_padded@f_variation[f_t_duplicate]
		smooth_f_b_result = W_b_padded@f_variation[f_b_duplicate]

		fit_norm_result = cp.square(cp.norm(elementwise_result, 2))

		# Smooth each branch separately and weigh by the proportional length of the
		# branch relative to the bottom branch (the longest)

		# Recovery branch is the shortest 0.9
		# Top branch: 1.0
		# Daughter branch: 1.2
		smooth_result = (cp.sum(cp.abs(smooth_f_i_result)) * 1 +
						 cp.sum(cp.abs(smooth_f_t_result)) * 1 +
						 cp.sum(cp.abs(smooth_f_b_result)) * 1)

		# todo: These weights may be too low for the bottom branch, and introduces the
		#       DG1 bias which depicts a greater amount of variability.

		kappa = self.kappa

		tb_regularization_result = f_variation[f_dg1] - f_variation[f_cg1]

		# L2 norm
		# cg1_dg1_regularization_result = cp.square(cp.norm(tb_regularization_result, 2))

		# L1 norm
		cg1_dg1_regularization_result = cp.sum(cp.abs(tb_regularization_result))

		objective = cp.Minimize(

			# L2 fitting norm
			fit_norm_result + 
			self.gamma * smooth_result +
			self.kappa * cg1_dg1_regularization_result
		)

		# Constraint for halted cells, non-negativity, and upper bounds to improve speed
		constraints = [f_variation >= 0, f_baseline >= 0, # non-negativity
			f_variation[f_i[0]] == f_variation[f_t[-1]+1], # halted cells
		]

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=cp.MOSEK)

		# Convert it into a numpy array
		f = f_variation.value+f_baseline.value

		self.f = f
		self.f_variation = f_variation.value

		self.tb_regularization_result = cg1_dg1_regularization_result.value
		self.sn = smooth_result.value
		self.rn = fit_norm_result.value
		self.f = f

		if self.verbose:
			print("Fit norm:", self.rn)
			print("Smoothing norm:", self.sn)
			print("Initial smoothing result: ", np.sum(np.abs(smooth_f_i_result.value)))
			print("Top smoothing result: ", np.sum(np.abs(smooth_f_t_result.value)))
			print("Bottom smoothing result: ", np.sum(np.abs(smooth_f_b_result.value)))


	def plot_fit(self, plot_timepoints=False):

		from matplotlib import pyplot as plt

		config = self.config
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		i_tps = config.get_timepoints_for_branch('i')
		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')

		if not plot_timepoints:
			# Plot by indices
			i_tps = np.arange(len(i_tps))
			t_tps = np.arange(len(t_tps))
			b_tps = np.arange(len(b_tps))

		num_cols = 4

		fig, axs = plt.subplots(1, num_cols, figsize=(16, 3))

		g = self.g

		f = self.f

		max_value = np.concatenate([g, f]).max()
		ylims = -((max_value*0.05)), (max_value*1.05)

		gamma_predicted_g = self.H@f

		ax_row = axs

		ax = ax_row[0]

		ax.plot(g[:], c='black', lw=3, label="Raw data")
		ax.plot(gamma_predicted_g, c='red',
				lw=3, label="Optimal $\\gamma$ solution")
		ax.set_title("Data vs Fit")
		ax.legend()
		ax.set_ylim(*ylims)

		ax = ax_row[1]
		ax.plot(i_tps, f[i_indices], c='red',
				lw=3)
		ax.set_title("Initial branch")
		ax.set_ylim(*ylims)

		ax = ax_row[2]
		ax.plot(b_tps, f[b_indices], c='blue',
				lw=3, alpha=0.25)
		ax.plot(t_tps, f[t_indices], c='red',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Top branch")

		ax = ax_row[3]
		ax.plot(t_tps, f[t_indices], c='red',
				lw=3, alpha=0.25)
		ax.plot(b_tps, f[b_indices], c='blue',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Bottom branch")
