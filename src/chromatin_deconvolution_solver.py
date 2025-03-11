
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


	def deconvolve_G_iteratively(self, gamma, verbose=False, verbose_progress=True,
		kappa=1e-4):
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
		self.kappa = kappa

		eps_cutoff = 1e-5
		deconvolution_solver = None

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
					N=self.N, f_replication=self.f_replication, b=self.b,
					kappa=kappa)

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

			if verbose and verbose_progress and i % 500 == 0:
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



def plot_branches(config, chrom, mnase_span, full_deconvolved_F, vmax=40, figsize=(5, 7)):

	from src.orf_plotter import load_default_orf_plotter
	from src.sgd import read_nondubious_genes_dataset

	orf_plotter = load_default_orf_plotter()
	orf_plotter.set_span_chrom(mnase_span, chrom)

	i_indices = config.get_Hpositions_for_branch('i')
	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')

	r_indices = config.get_Hpositions_for_phase('RG1')
	c_indices = config.get_Hpositions_for_phase('CG1')
	d_indices = config.get_Hpositions_for_phase('DG1')

	full_F_imgs = full_deconvolved_F.reshape((full_deconvolved_F.shape[0], 26, -1))
	full_F_imgs.shape

	num_imgs_per_branch = 12

	fig, axs = plt.subplots(num_imgs_per_branch+1, 4, figsize=figsize)
	axs = np.array(axs).T

	vmax_2 = vmax//2

	extent = [mnase_span[0], mnase_span[1], 0, 250]

	def plot_branch_imgs(row_axs, t_indices):

		for plot_index, index_in_t in enumerate(np.linspace(0,
			len(t_indices)-1, num_imgs_per_branch)):
			image_index = t_indices[int(index_in_t)]

			ax = row_axs[plot_index]
			ax.imshow(full_F_imgs[image_index], aspect='auto', cmap='magma_r', vmax=vmax,
					  origin='lower', extent=extent)
			ax.set_xticks([])
			ax.set_yticks([])
			
			
	def plot_difference(row_axs, t_indices, b_indices):

		eps = 1
		for plot_index, index_in_t in enumerate(np.linspace(0,
			len(t_indices)-1, num_imgs_per_branch)):
			
			image_index_t = t_indices[int(index_in_t)]
			image_index_b = b_indices[int(index_in_t)]

			diff_vmax_2 = 3
			img_diff = np.log2((full_F_imgs[image_index_b]+eps)/(full_F_imgs[image_index_t]+eps))


			# diff_vmax_2 = 30
			# img_diff = (full_F_imgs[image_index_b]) - (full_F_imgs[image_index_t])
			
			ax = row_axs[plot_index]
			ax.imshow(img_diff, aspect='auto', cmap='RdBu_r', vmin=-diff_vmax_2, vmax=diff_vmax_2,
					  origin='lower', extent=extent)
			ax.set_xticks([])
			ax.set_yticks([])

	axs[0][0].set_title("Recovery")
	axs[1][0].set_title("Mother")
	axs[2][0].set_title("Daughter")

	for ax in axs.T[0]:
		orf_plotter.plot_orf_annotations(ax)

	# Skip first row
	img_axs = (axs.T[1:]).T
			
	plot_branch_imgs(img_axs[0], i_indices)
	plot_branch_imgs(img_axs[1], t_indices)
	plot_branch_imgs(img_axs[2], b_indices)

	plot_difference(img_axs[3], t_indices, b_indices)

	plt.suptitle("Deconvolved chromatin")
	plt.subplots_adjust(top=0.9, hspace=0, wspace=0.05)

	return fig


def subset_select_highest_G_indices(G, num_examples=5):
    G_max_df = pd.DataFrame({'max_value': G.max(axis=0), 
        'index': np.arange(G.shape[1])})
    highest_Gs = G_max_df.sort_values('max_value', 
        ascending=False).head(num_examples)['index'].values
    G_values = G[:, highest_Gs].copy()
    return G_values, highest_Gs


def dummy_N_frep_b(config):
	"""Example N, f-replication and b for testing, serves as a stand-in for 
	copy correction specifications. Will be replaced by actual replication timing
	profile."""

	n, m = config.H.shape

	b = 1

	# Example normalization term from chromosome 4
	chr4_n = np.array([1.00442364, 1.00347336, 0.97773201, 0.8061213 , 0.66388052,
		   0.84322967, 0.99955218, 1.00893152, 0.97884289, 0.85742082,
		   0.75599297, 0.76982516, 0.84182678, 0.97059662, 0.99078551,
		   0.91119956])

	s_indices = config.get_Hpositions_for_phase('S')
	repl_idx = s_indices[len(s_indices)//2]

	N = np.diag(chr4_n)
	f_rep = np.ones(m)
	f_rep[repl_idx:-1] =  2

	return N, f_rep, b


def dummy_no_copy_correction_N_frep_b(config):
	"""Example N, f-replication and b for testing, serves as a stand-in for 
	copy correction specifications. Will be replaced by actual replication timing
	profile."""

	n, m = config.H.shape

	b = 1

	# Example normalization term from chromosome 4
	n_diag = np.ones(n)

	N = np.diag(n_diag)
	f_rep = np.ones(m)

	return N, f_rep, b


def plot_example_fits(chromatin_solver):

	config = chromatin_solver.config
	i_indices = config.get_Hpositions_for_branch('i')
	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')

	num_examples = 5
	num_cols = 4

	fig, axs = plt.subplots(num_examples, num_cols, figsize=(23, 11))

	F_gamma_solution = chromatin_solver.F
	G = chromatin_solver.G

	gamma_predicted_G = chromatin_solver.compute_predicted_G(F_gamma_solution)

	# Select bins with the highest max values
	import pandas as pd
	G_max_df = pd.DataFrame({'max_value': G.max(axis=0), 'index': np.arange(G.shape[1])})
	highest_Gs = G_max_df.sort_values('max_value', ascending=False).head(num_examples)['index'].values

	highest_g = G_max_df.max_value.max()
	highest_f = F_gamma_solution[:, highest_Gs].max()
	highest_val = max(highest_g, highest_f)

	ylims = -highest_val*0.05, highest_val*1.05

	for i in range(num_examples):

		ax_row = axs[i]

		if i != num_examples-1:
			for ax in ax_row:
				ax.set_xticks([])
				ax.set_yticks([])

		f_bin_index = highest_Gs[i]

		ax = ax_row[0]

		ax.plot(G[:, f_bin_index], c='black', lw=3, label="Raw data")
		ax.plot(gamma_predicted_G[:, f_bin_index].T, c='red',
				lw=3, label="Optimal $\\gamma$ solution")
		if i == 0: 
			ax.set_title("Data vs Fit")
			ax.legend()
		ax.set_ylim(*ylims)

		ax = ax_row[1]
		ax.plot(F_gamma_solution[b_indices, f_bin_index].T, c='blue',
				lw=3, alpha=0.25)
		ax.plot(F_gamma_solution[t_indices, f_bin_index].T, c='red',
				lw=3, alpha=0.25)
		ax.plot(F_gamma_solution[i_indices, f_bin_index].T, c='black',lw=3)
		if i == 0:
			ax.set_title("Initial branch")
		ax.set_ylim(*ylims)

		ax = ax_row[2]
		ax.plot(F_gamma_solution[b_indices, f_bin_index].T, c='blue',
				lw=3, alpha=0.25)
		ax.plot(F_gamma_solution[t_indices, f_bin_index].T, c='red',
				lw=3)
		ax.set_ylim(*ylims)
		if i == 0: ax.set_title("Top branch")

		ax = ax_row[3]
		ax.plot(F_gamma_solution[t_indices, f_bin_index].T, c='red',
				lw=3, alpha=0.25)
		ax.plot(F_gamma_solution[b_indices, f_bin_index].T, c='blue',
				lw=3)

		ax.set_ylim(*ylims)
		if i == 0: ax.set_title("Bottom branch")

	return fig

