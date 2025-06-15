import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel
from matplotlib.colors import ListedColormap
from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver


class ChromatinFindOptimalGamma(object):
	"""Wrapper to find optimal gamma for a window of chromatin reads"""

	def __init__(self, chromatin_solver, gamma_min=1e-2, gamma_max=10,
		verbose=True, kappa=1.0, eta=0):

		# Refactoring of the find optimal gamma code
		from src.find_gamma_refactor import GammaOptimizer
			
		def compute_solution(gamma_value):
			"""Function to compute the solution, rn, and sn for the optimizer"""
			F = chromatin_solver.deconvolve_G_iteratively(gamma=gamma_value,
														 verbose=False,
														 verbose_progress=False,
														 kappa=kappa,
														 eta=eta)

			# Average residual norm for each bin
			rn = chromatin_solver.rn

			# Average smoothing norm for each bin
			sn = chromatin_solver.sn

			# -- Computation of the smoothing norm over the total occupancy of the solution --
			mean_F = F.mean(1)
			mean_G = chromatin_solver.G.mean(1)

			# Take the wavelet W_i (identical to t and b)
			deconvolution_solver = chromatin_solver.deconvolution_solver
			wavelet_mat = deconvolution_solver.W_i

			# Compute the smoothing norm on the mean_F as an auxiliary term
			from src.deconvolution_solver import compute_smoothing_result

			mean_sn = compute_smoothing_result(mean_F, 
				deconvolution_solver.f_i_mirror, 
				deconvolution_solver.f_t_periodic,
				deconvolution_solver.f_b_periodic,
				wavelet_mat, wavelet_mat, wavelet_mat, is_cvxpy=False)

			# ---------------------------------------------------------------------------------

			return_dictionary = {
				'solution_F': F,
				'sn': sn,
				'rn': rn,
				'mean_sn': mean_sn
			}

			return return_dictionary

		gamma_optimizer = GammaOptimizer(compute_solution, gamma_min=gamma_min, gamma_max=gamma_max,
										 verbose=verbose, mode='chromatin',
										 early_stop_enabled=True)

		self.chromatin_solver = chromatin_solver
		self.gamma_optimizer = gamma_optimizer

	def find_optimal_gamma(self, plot=True):

		from src.timer import Timer
		timer = Timer()
		self.gamma_optimizer.calculate_base_error()
		self.gamma_optimizer.calculate_error_boundaries()

		#self.gamma_optimizer.find_boundary_gammas()

		self.gamma_optimizer.gamma_left = 0.01
		self.gamma_optimizer.gamma_right = 10.0
		print(f"Using manual left and right gamma settings: {self.gamma_optimizer.gamma_left} - "
			 f"{self.gamma_optimizer.gamma_right}")

		self.gamma_optimizer.find_elbow()

		if plot:
			fig = self.gamma_optimizer.plot_elbow()

		timer.print_time("Completed.")

		return self.gamma_optimizer.optimal_gamma


	def plot_gamma_sweep(self):

		gamma_optimizer = self.gamma_optimizer
		optimal_solution_index = int(gamma_optimizer.elbow_results_df.loc[gamma_optimizer.optimal_gamma].solution_index)

		config = self.chromatin_solver.config
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		num_examples = 5
		num_cols = 4

		fig, axs = plt.subplots(num_examples, num_cols, figsize=(16, 11))

		gamma_sweep = self.gamma_optimizer.elbow_results_df.index
		F_gamma_solutions = self.gamma_optimizer.elbow_solutions
		G = self.chromatin_solver.G

		chromatin_solver = self.chromatin_solver
		gamma_predicted_Gs = np.array([chromatin_solver.compute_predicted_G(F_gamma_solutions[i]) 
		 for i in range(F_gamma_solutions.shape[0])])

		cmap = ListedColormap(plt.cm.viridis(np.linspace(0.2, 0.8, 256)))

		# Select bins with the highest max values
		import pandas as pd
		G_max_df = pd.DataFrame({'max_value': G.max(axis=0), 'index': np.arange(G.shape[1])})
		highest_Gs = G_max_df.sort_values('max_value', ascending=False).head(num_examples)['index'].values

		highest_g = G_max_df.max_value.max()
		ylims = -highest_g*0.05, highest_g*1.5

		for i in range(num_examples):

			if i >= len(highest_Gs): break

			ax_row = axs[i]

			if i != num_examples-1:
				for ax in ax_row:
					ax.set_xticks([])
					ax.set_yticks([])

			f_bin_index = highest_Gs[i]

			ax = ax_row[0]

			for j in range(gamma_predicted_Gs.shape[0]):
				ax.plot(gamma_predicted_Gs[j, :, f_bin_index].T, c=cmap(j/len(gamma_predicted_Gs)))

			sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gamma_sweep.min(),
				vmax=gamma_sweep.max()))
			cbar = plt.colorbar(sm, ax=ax)
			cbar.ax.set_ylabel('Smoothness, $\\gamma$', rotation=270, va='bottom')

			ax.plot(G[:, f_bin_index], c='black', lw=3, label="Raw data")
			ax.plot(gamma_predicted_Gs[optimal_solution_index, :, f_bin_index].T, c='red',
					lw=3, label="Optimal $\\gamma$ solution")
			if i == 0: 
				ax.set_title("Data vs Fit")
				ax.legend()
			ax.set_ylim(*ylims)

			ax = ax_row[1]
			for j in range(F_gamma_solutions.shape[0]):
				ax.plot(F_gamma_solutions[j, i_indices, f_bin_index].T, c=cmap(j/len(F_gamma_solutions)))

			ax.plot(F_gamma_solutions[optimal_solution_index, i_indices, f_bin_index].T, c='red',
					lw=3)
			if i == 0:
				ax.set_title("Initial branch")
			ax.set_ylim(*ylims)

			ax = ax_row[2]
			for j in range(F_gamma_solutions.shape[0]):
				ax.plot(F_gamma_solutions[j, t_indices, f_bin_index].T, c=cmap(j/len(F_gamma_solutions)))
			ax.plot(F_gamma_solutions[optimal_solution_index, t_indices, f_bin_index].T, c='red',
					lw=3)
			ax.set_ylim(*ylims)
			if i == 0: ax.set_title("Top branch")

			ax = ax_row[3]
			for j in range(F_gamma_solutions.shape[0]):
				ax.plot(F_gamma_solutions[j, b_indices, f_bin_index].T, c=cmap(j/len(F_gamma_solutions)))
			ax.plot(F_gamma_solutions[optimal_solution_index, b_indices, f_bin_index].T, c='red',
					lw=3)
			ax.set_ylim(*ylims)
			if i == 0: ax.set_title("Bottom branch")

		return gamma_predicted_Gs


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
		prediction = N@config.H@np.multiply(Fs[i], f_rep)
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

def find_gamma_chromatin(chromatin_save_directory, window_set_path, index,
	copy_correct=True, kappa=1, eta=0):

	from src.utils import mkdirs_safe, parse_bool, print_fl
	from src.combined_chromatin_model import CombinedChromatinModel
	from src.chromatin_model import ChromatinModel
	from src.config import load_default_chrom_configs, load_cloccs_configs

	# Load the configs from disk
	config1, config2 = load_default_chrom_configs()

	def deconv_and_save(chrom, mnase_span, chromatin_save_directory):

		data_directory = f"{chromatin_save_directory}/deconvolution_data/chr{chrom}"
		raw_plots_directory = f"{chromatin_save_directory}/raw_plots_directory/chr{chrom}"
		deconv_plots_directory = f"{chromatin_save_directory}/deconv_plots_directory/chr{chrom}"

		mkdirs_safe([data_directory, raw_plots_directory, deconv_plots_directory])
		combined_model = CombinedChromatinModel(config1=config1, config2=config2)

		# Load window to deconvolve
		combined_model.load_mnase_span(chrom, mnase_span)

		save_title = f"chr{chrom}_{mnase_span[0]}_{mnase_span[1]}"

		# Plot normalization check
		fig = combined_model.plot_normalization_sanity_check()
		plt.savefig(f"{raw_plots_directory}/normalization_{save_title}.png")
		plt.close(fig)

		# Plot raw data
		fig = combined_model.chrom1_model.plot_raw_data(figsize=(11, 11))
		plt.savefig(f"{raw_plots_directory}/raw_rep1_{save_title}.png")
		plt.close(fig)

		fig = combined_model.chrom2_model.plot_raw_data(figsize=(11, 11))
		plt.savefig(f"{raw_plots_directory}/raw_rep2_{save_title}.png")
		plt.close(fig)

		combined_model.setup_deconv_model(copy_correct=copy_correct)

		# Find optimum run, just save gamma and elbow
		combined_model.deconvolve_find_optimal_gamma(kappa=kappa, eta=eta,
			verbose=True)	

		# Save the gamma and elbow curve to disk
		save_gamma_path = f"{data_directory}/{save_title}_gamma.txt"
		with open(save_gamma_path, 'w') as save_file:
			save_file.write(f"{combined_model.gamma}")

		save_elbow_path = f"{deconv_plots_directory}/{save_title}_elbow.png"
		combined_model.find_gamma_chromatin.plot_gamma_sweep()
		plt.savefig(save_elbow_path)

	# Read the window datas data and load the relevant row
	window_set = pd.read_csv(window_set_path)
	row = window_set.iloc[index]

	chrom = row.chr
	span = row.start, row.end+1 # (Add 1 to include the last base)

	print_fl(f"Deconvolving index:{index}, chr{chrom}, {span[0], span[1]}")

	deconv_and_save(chrom, span, chromatin_save_directory)
