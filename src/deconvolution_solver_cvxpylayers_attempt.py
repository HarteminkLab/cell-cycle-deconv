import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from src.helpers import get_wavelet_kernel

class DeconvolutionSolver(object):

	def __init__(self, config, g, H, gamma, N=None, f_replication=None,
		b=None, padding_type='both', obj_error_mode='additive', kappa=5e-3,
		data_is_logged=True, unlog_transform=False, log_transform=False,
		use_gpu=False, verbose=False):

		n, m = H.shape

		self.verbose = verbose

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
		self.use_gpu = use_gpu
		
		self.gamma = gamma
		self.padding_type = padding_type
	
	def deconvolve(self):

		# Set up all the problem components as before
		f_i = self.config.get_Hpositions_for_branch('i')
		f_t = self.config.get_Hpositions_for_branch('t')
		f_b = self.config.get_Hpositions_for_branch('b')
		f_cg1 = self.config.get_Hpositions_for_phase('CG1')
		f_rg1 = self.config.get_Hpositions_for_phase('RG1')
		f_dg1 = self.config.get_Hpositions_for_phase('DG1')
		f_postg1 = self.config.get_Hpositions_for_phase('postG1')
		f_halted = self.config.get_Hpositions_for_phase('Halted')

		# The left end of the recovery branch is mirrored, the right end is periodically
		# smooth into the start of the top branch
		branch_len_itb = len(f_i) # assume each branch is the same length
		half_branch_len_itb = branch_len_itb // 2

		f_i_mirror = np.concatenate([np.flip(f_i), f_i])
		f_t_periodic = np.concatenate([f_t, f_t])
		f_b_periodic = np.concatenate([f_b, f_b])

		enable_padding = False

		# Seems like padding may not be necessary, so we'll set these all to zero
		if enable_padding:
			padding_left_t = branch_len_itb # Start of MG1 padding
			padding_left_b = branch_len_itb # Start of DG1 padding
			padding_right = branch_len_itb # End of post G1 padding
		else:
			padding_left_t = 0
			padding_left_b = 0
			padding_right = 0

		total_padding = padding_left_t+padding_left_b+padding_right

		# Convex optimization
		n, m = self.H.shape

		f_indices = np.arange(m)
		f_variation_padded = cp.Variable(m+total_padding)

		# Define the indices in the full f vector for the padding
		f_padding_left_t_indices = np.arange(m, m+padding_left_t)
		f_padding_left_b_indices = np.arange(m+padding_left_t, 
			m+padding_left_t+padding_left_b)
		f_padding_right_indices = np.arange(m+padding_left_t+padding_left_b, 
			m+padding_left_t+padding_left_b+padding_right)

		# Mirrorred start means we can use the flipped post G1 padding
		f_padded_i = np.concatenate([np.flip(f_padding_right_indices), 
									 f_i_mirror,
									 f_padding_right_indices])

		f_padded_t = np.concatenate([f_padding_left_t_indices, 
									 f_t_periodic,
									 f_padding_right_indices])

		f_padded_b = np.concatenate([f_padding_left_b_indices, 
									 f_b_periodic,
									 f_padding_right_indices])

		# Create block matrix structures for wavelets
		W_i = get_wavelet_kernel(len(f_padded_i), par=5)
		W_t = get_wavelet_kernel(len(f_padded_t), par=5)
		W_b = get_wavelet_kernel(len(f_padded_b), par=5)

		f_variation = f_variation_padded[f_indices]

		# Model a baseline value, so smoothing constraints are applied to
		# variations on the baseline
		f_baseline = cp.Variable(1)

		H = self.H
		g = self.g
		N = self.N
		b = self.b

		def compute_fit_result(N, H, f_variation, f_baseline, b,
			is_cvxpy=True):

			if is_cvxpy:
				norm_func = cp.norm
				square_func = cp.square
				mult_func = cp.multiply
			else:
				norm_func = np.linalg.norm
				square_func = np.square
				mult_func = np.multiply

			f_non_replicative = f_variation+f_baseline
			f_replication = self.f_replication
			f_combined = mult_func(f_non_replicative, f_replication)

			if self.obj_error_mode == 'multiplicative':
				eps = 1e-5
				elementwise_result = (N@H@f_combined*b)/(g+eps) - 1
			elif self.obj_error_mode == 'additive':
				elementwise_result = N@H@f_combined*b - g
			else:
				raise ValueError(f"Unimplemented objective error mode: {self.obj_error_mode}")

			fit_norm_result = square_func(norm_func(elementwise_result, 2))
			return fit_norm_result

		def compute_smoothing_result(f_variation_padded, f_padded_i, f_padded_t, f_padded_b,
			W_i, W_t, W_b, is_cvxpy):

			if is_cvxpy:
				sum_func = cp.sum
				abs_func = cp.abs
			else:
				sum_func = np.sum
				abs_func = np.abs

			coeffs_i = W_i@(f_variation_padded[f_padded_i])
			coeffs_t = W_t@(f_variation_padded[f_padded_t])
			coeffs_b = W_b@(f_variation_padded[f_padded_b])

			smooth_f_i_result = sum_func(abs_func(coeffs_i))
			smooth_f_t_result = sum_func(abs_func(coeffs_t))
			smooth_f_b_result = sum_func(abs_func(coeffs_b))

			smooth_result = (smooth_f_i_result * 2 +
							 smooth_f_t_result * 1 +
							 smooth_f_b_result * 1)

			return smooth_result

		fit_norm_result = compute_fit_result(N, H, f_variation, f_baseline, b, True)
		smooth_result = compute_smoothing_result(f_variation_padded, 
			f_padded_i, f_padded_t, f_padded_b,
			W_i, W_t, W_b, True)

		kappa = self.kappa

		tb_regularization_result = f_variation[f_dg1] - f_variation[f_cg1]

		# L1 norm
		cg1_dg1_regularization_result = cp.sum(cp.abs(tb_regularization_result))

		# Create a dummy parameter for CVXPYLayers (since all other values are constants)
		dummy_param = cp.Parameter(nonneg=True)
		dummy_param.value = 1.0

		objective = cp.Minimize(
			# L2 fitting norm
			fit_norm_result + 
			self.gamma * smooth_result +
			self.kappa * cg1_dg1_regularization_result +
			0 * dummy_param  # Add dummy parameter with zero coefficient
		)

		# Constraint for halted cells, non-negativity, and upper bounds to improve speed
		constraints = [f_variation >= 0, f_baseline == 0] # non-negativity

		prob = cp.Problem(objective, constraints)
		
		# Use CVXPYLayers with GPU acceleration if requested
		if self.use_gpu:
			# Create the layer with our dummy parameter
			layer = CvxpyLayer(prob, 
							 parameters=[dummy_param], 
							 variables=[f_variation_padded, f_baseline])
			
			# Determine device (GPU or CPU)
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			if device.type == 'cpu' and self.use_gpu:
				print("Warning: GPU requested but not available. Using CPU instead.")
			
			# Create the dummy tensor and move to device
			dummy_tensor = torch.tensor([1.0], dtype=torch.float32).to(device)
			
			# Solve the problem with GPU acceleration
			f_variation_padded_torch, f_baseline_torch = layer(dummy_tensor,
				solver_args={"solve_method": "clarabel"})	

			# Convert PyTorch tensors back to numpy arrays
			f_variation_padded_value = f_variation_padded_torch.detach().cpu().numpy()[0]
			f_baseline_value = f_baseline_torch.detach().cpu().numpy()[0]
			
			# Extract the relevant portion of the solution
			f_variation_value = f_variation_padded_value[f_indices]
			
		else:
			# Use traditional CVXPY solving with MOSEK or other solver
			result = prob.solve(solver=cp.MOSEK)
			
			# Extract solution values
			f_variation_padded_value = f_variation_padded.value
			f_baseline_value = f_baseline.value
			f_variation_value = f_variation.value
		
		# Store the results
		self.f = f_variation_value + f_baseline_value
		self.f_variation = f_variation_value
		self.f_full = f_variation_padded_value
		
		# Store the other optimization results
		self.rn = compute_fit_result(N, H, f_variation_value, f_baseline_value, b, False)
		self.sn = compute_smoothing_result(f_variation_padded_value, 
			f_padded_i, f_padded_t, f_padded_b,
			W_i, W_t, W_b, False)
		
		# Store processed data for later use
		self.f_i_mirror = f_i_mirror
		self.f_t_periodic = f_t_periodic
		self.f_b_periodic = f_b_periodic
		self.W_i = W_i
		
		if self.verbose:
			print("Fit norm:", self.rn)
			print("Smoothing norm:", self.sn)
			if self.use_gpu:
				print("Solved using CVXPYLayers")
		
		return self.f

	def plot_fit(self, plot_timepoints=True):

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
