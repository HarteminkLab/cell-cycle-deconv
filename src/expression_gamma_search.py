import numpy as np
from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel
from matplotlib.colors import ListedColormap
from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver


class GeneExpressionFindOptimalGamma(object):
	"""Wrapper to find optimal gamma for a window of gene expression reads"""

	def __init__(self, config, H, gene_expression, gamma_min=0.001, gamma_max=1, verbose=True):

		from src.deconvolution_solver import DeconvolutionSolver

		# config will be used for indices, so the combined model can use either replicate's config
		# for the combined model, gene expressn and H are assumed to be concatenated properly
		deconvolution_solver = DeconvolutionSolver(config, g=gene_expression, H=H, gamma=0.0,
		                                          obj_error_mode='additive', use_gpu=True)

		# Refactoring of the find optimal gamma code
		from src.find_gamma_refactor import GammaOptimizer
		self.H = H
		self.config = config
			
		def compute_solution(gamma_value):
			"""Function to compute the solution, rn, and sn for the optimizer"""
			deconvolution_solver.gamma = gamma_value
			deconvolution_solver.deconvolve()
			rn = deconvolution_solver.rn
			sn = deconvolution_solver.sn
			f = deconvolution_solver.f
			return f, sn, rn

		gamma_optimizer = GammaOptimizer(compute_solution, gamma_min=gamma_min, gamma_max=gamma_max,
										 verbose=verbose)

		self.deconvolution_solver = deconvolution_solver
		self.gamma_optimizer = gamma_optimizer

	def find_optimal_gamma(self, plot=True, verbose=True, kappa=None):

		from src.timer import Timer
		timer = Timer()

		if kappa is not None:
			self.deconvolution_solver.kappa = kappa

		self.gamma_optimizer.verbose = verbose
		self.gamma_optimizer.calculate_base_error()

		# todo: finding the left and right gamma limits using 
		#       error boundaries no longer produces consistent results, likely
		#       due to changes in the original guo model, reverting to 
		#       a sweep through the minimum and maximum gamma values
		#
		self.gamma_optimizer.calculate_error_boundaries()
		self.gamma_optimizer.find_boundary_gammas()

		# self.gamma_optimizer.gamma_left = self.gamma_optimizer.gamma_min
		# self.gamma_optimizer.gamma_right = self.gamma_optimizer.gamma_max

		self.gamma_optimizer.find_elbow()

		if plot:
			self.gamma_optimizer.plot_elbow()

		if verbose:
			timer.print_time("Completed.")

	def retrieve_solution(self):
		optimal_solution_index = int(self.gamma_optimizer.elbow_results_df.loc[self.gamma_optimizer.optimal_gamma].solution_index)
		self.optimal_F = self.gamma_optimizer.elbow_solutions[optimal_solution_index]
		return self.optimal_F

	def plot_gamma_sweep(self, plot_log_transform=True):

		gamma_optimizer = self.gamma_optimizer
		optimal_solution_index = int(gamma_optimizer.elbow_results_df.loc[gamma_optimizer.optimal_gamma].solution_index)

		config = self.deconvolution_solver.config
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		i_tps = config.get_timepoints_for_branch('i')
		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')

		num_cols = 4

		fig, axs = plt.subplots(1, num_cols, figsize=(16, 3))

		gamma_sweep = self.gamma_optimizer.elbow_results_df.index
		f_gamma_solutions = self.gamma_optimizer.elbow_solutions
		g = self.deconvolution_solver.g

		deconvolution_solver = self.deconvolution_solver

		def compute_predicted_g(f):
			return self.H@f

		gamma_predicted_gs = np.array([compute_predicted_g(f_gamma_solutions[i]) 
			for i in range(f_gamma_solutions.shape[0])])

		if not plot_log_transform:
			g = 2**g
			f_gamma_solutions = 2**f_gamma_solutions
			gamma_predicted_gs = 2**gamma_predicted_gs

		g_vmax = g.max()
		vmax = max(g.max(), f_gamma_solutions[optimal_solution_index].max())
		ylims = (vmax*-0.05, vmax*1.2)
		g_ylims = (g_vmax*-0.05, g_vmax*1.2)

		if g.max() == 0:
			ylims = -0.1, 1

		cmap = ListedColormap(plt.cm.viridis(np.linspace(0.2, 0.8, 256)))

		ax_row = axs

		ax = ax_row[0]

		timepoints = config.timepoints

		for j in range(gamma_predicted_gs.shape[0]):
			ax.plot(timepoints, gamma_predicted_gs[j, :].T, 
				c=cmap(j/len(gamma_predicted_gs)))

		sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gamma_sweep.min(),
			vmax=gamma_sweep.max()))
		cbar = plt.colorbar(sm, ax=ax)
		cbar.ax.set_ylabel('Smoothness, $\\gamma$', rotation=270, va='bottom')

		ax.plot(timepoints, 
			g[:], c='black', lw=3, label="Raw data")
		ax.plot(timepoints, 
			gamma_predicted_gs[optimal_solution_index, :].T, c='red',
				lw=3, label="Optimal $\\gamma$ solution")
		
		ax.set_ylim(*g_ylims)
		ax.set_title("Data vs Fit")
		ax.legend()

		ax = ax_row[1]
		for j in range(f_gamma_solutions.shape[0]):
			ax.plot(i_tps, f_gamma_solutions[j, i_indices].T, c=cmap(j/len(f_gamma_solutions)))

		ax.plot(i_tps, f_gamma_solutions[optimal_solution_index, i_indices].T, c='red',
				lw=3)
		ax.set_title("Initial branch")
		ax.set_ylim(*ylims)

		ax = ax_row[2]
		for j in range(f_gamma_solutions.shape[0]):
			ax.plot(t_tps, f_gamma_solutions[j, t_indices].T, c=cmap(j/len(f_gamma_solutions)))
		ax.plot(t_tps, f_gamma_solutions[optimal_solution_index, t_indices].T, c='red',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Top branch")

		ax = ax_row[3]
		for j in range(f_gamma_solutions.shape[0]):
			ax.plot(b_tps, f_gamma_solutions[j, b_indices].T, c=cmap(j/len(f_gamma_solutions)))
		ax.plot(b_tps, f_gamma_solutions[optimal_solution_index, b_indices].T, c='red',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Bottom branch")

		plt.suptitle(f"$\\gamma$ = {self.gamma_optimizer.optimal_gamma:.4g}", y=1.1, fontsize=16, fontweight='demi')

		return fig


def create_gamma_sweep_plots_single_measure(config, N, H, Fs, f_rep, gamma_sweep, G,
	ylims=(0, 1)):

	i_indices = config.get_Hpositions_for_branch('i')
	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')

	i_timepoints = config.get_timepoints_for_branch('i')
	t_timepoints = config.get_timepoints_for_branch('t')

	plt.figure(figsize=(11, 6))
	plt.subplot(2, 2, 1)

	tb_indices = np.concatenate([t_indices, b_indices])

	W_i = get_wavelet_kernel(len(i_indices))
	W_tb = get_wavelet_kernel(len(tb_indices))

	cmap = ListedColormap(plt.cm.inferno(np.linspace(0.25, 0.85, 256)))

	for i in range(Fs.shape[0]):
		F = Fs[i]
		smoothness_i = W_i@F[i_indices]
		plt.plot(smoothness_i, c=cmap(i/len(Fs)))

	for i in range(Fs.shape[0]):
		F = Fs[i]
		smoothness_tb = W_tb@F[tb_indices]
		plt.plot(smoothness_tb, c=cmap(i/len(Fs)))

	plt.title("Wavelet coefficients")
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gamma_sweep.min(),
		vmax=gamma_sweep.max()))
	W_len = W_i.shape[0]
	plt.xticks([0.15*W_len, 0.85*W_len], ["Lower frequency", "Higher frequency"])
	plt.xlim(0, W_len)
	plt.gca().tick_params(axis='x', length=0)
	plt.ylim(-0.4, 0.4)

	plt.subplot(2, 2, 2)
	for i in range(Fs.shape[0]):
		prediction = N@self.H@np.multiply(Fs[i], f_rep)
		plt.plot(config.timepoints, prediction, c=cmap(i/len(Fs)))
	plt.plot(config.timepoints, G, c='black', lw=4, label="Raw data")
	plt.legend()
	plt.title("Goodness of fit")
	plt.ylim(*ylims)
	plt.xlabel("Experiment time, min")
	plt.xticks(np.arange(0, config.timepoints[-1], 20))
	plt.xlim(0, config.timepoints[-1])

	cbar = plt.colorbar(sm)
	cbar.ax.set_ylabel('Smoothness, $\\gamma$', rotation=270, va='bottom')

	plt.subplot(2, 2, 3)
	for i in range(Fs.shape[0]):
		plt.plot(i_timepoints, Fs[i, i_indices], c=cmap(i/len(Fs)))
	plt.title("Initial branch")
	plt.ylim(*ylims)
	plt.xlim(0, 60)
	plt.xlabel("Average single cell cycle time, min")

	plt.subplot(2, 2, 4)
	for i in range(Fs.shape[0]):
		plt.plot(t_timepoints, Fs[i, t_indices], c=cmap(i/len(Fs)))
	plt.title("Top branch")
	plt.ylim(*ylims)
	plt.xlim(0, 60)
	plt.xlabel("Average single cell cycle time, min")

	plt.subplots_adjust(hspace=0.6, top=0.86)

	plt.suptitle("Gamma sweep of single chromatin metric", fontsize=16)


def plot_coefficients_expression(expression_find_gamma):
	fig, axs = plt.subplots(2, 3, figsize=(13, 5))

	def plot_coefficients(branch, ax0, ax1):
	    
	    if branch == 'b':
	        f_padded_bottom_indices = expression_find_gamma.deconvolution_solver.f_bottom_padded_indices
	    elif branch == 'i':
	        f_padded_bottom_indices = expression_find_gamma.deconvolution_solver.f_recovery_padded_indices
	    elif branch == 't':
	        f_padded_bottom_indices = expression_find_gamma.deconvolution_solver.f_top_padded_indices

	    weights_coeffs = expression_find_gamma.deconvolution_solver.coefficient_weights_itb
	    W = expression_find_gamma.deconvolution_solver.W_itb
	    f_padded = expression_find_gamma.deconvolution_solver.f_padded
	    coefficients = W@f_padded[f_padded_bottom_indices]
	    weighted_coefficients = coefficients * weights_coeffs

	    ax0.plot(coefficients)
	    ax1.plot(weighted_coefficients)

	plot_coefficients('i', axs[0][0], axs[1][0])
	plot_coefficients('t', axs[0][1], axs[1][1])
	plot_coefficients('b', axs[0][2], axs[1][2])
