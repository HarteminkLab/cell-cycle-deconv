
import numpy as np
import matplotlib.pyplot as plt


def get_dg1_bias_scale(kappa, temporal_ratio, mode='log'):
	"""As a function of kappa, as we approach higher values of kappa

	we turn the bias corrector off at varying rates

	kappa: 0-1
	temporal ratio > 1.0

	returns a scalar between [temporal_ratio, 1.0]
	"""

	from collections.abc import Iterable

	# Do not exceed 1
	if not isinstance(kappa, Iterable):
		if kappa > 1: kappa = 1
	else:
		kappa[kappa > 1] = 1

	if mode == 'linear':
		beta = kappa
	elif mode == 'quadratic':
		beta = kappa**2
	elif mode == 'sqrt':
		beta = np.sqrt(kappa)
	elif mode == 'fourthrt':
		beta = kappa**(1./4)
	elif mode == '16thrt':
		beta = (kappa)**(1./16)
	elif mode == 'log':

		# Linear mapping in log space
		# Assume the min and max thresholds for kappa,
		# empirically these are the boundaries for kappa
		min_kappa_thresh = 1e-4
		max_kappa_thresh = 1

		neg_log_min_kap = -np.log10(min_kappa_thresh)
		beta = (np.log10(kappa)+neg_log_min_kap)*(1./neg_log_min_kap)

		if not isinstance(kappa, Iterable):
			beta = min(max(beta, 0), 1)
		else:
			beta[beta < 0] = 0
			beta[beta > 1] = 1

	elif mode == 'constant':
		if not isinstance(kappa, Iterable):
			beta = 0
		else:
			beta = np.zeros_like(kappa)
	elif mode == None or mode == 'None':
		if not isinstance(kappa, Iterable):
			beta = 1
		else:
			beta = np.ones_like(kappa)
	else:
		raise ValueError("Unhandled mode", mode)

	return temporal_ratio - beta * (temporal_ratio - 1)

def plot_phi_curves_comparison():
	kappa_min, kappa_max = 1e-4, 1

	thresh_ratio = 1.4

	logkappas = np.linspace(np.log10(kappa_min), np.log10(kappa_max), 100)
	kappas = 10**logkappas

	plt.figure(figsize=(8, 3))

	def plot_kappas_mode(kappas, mode):
		plt.plot(kappas, get_dg1_bias_scale(kappas, thresh_ratio, mode=mode), label=mode)

	def plot_set():
		plot_kappas_mode(kappas, 'constant')
		plot_kappas_mode(kappas, 'linear')
		plot_kappas_mode(kappas, 'fourthrt')
		plot_kappas_mode(kappas, 'log')
		plot_kappas_mode(kappas, 'None')
		plt.xlabel("$\\kappa$")
		plt.ylabel("DG1 bias scalar")
		plt.legend(loc='lower left')

	plt.subplot(1, 2, 1)
	plot_set()

	plt.subplot(1, 2, 2)
	plot_set()
	plt.xscale('log')

	plt.suptitle("DG1 bias scalar function")



def plot_kappas_for_fits(config, F_kappa_curves, G, kappa_values):

	from src.chromatin_deconvolution_solver import dummy_N_frep_b, \
		subset_select_highest_G_indices

	i_indices = config.get_Hpositions_for_branch('i')
	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')

	num_examples = 5
	num_cols = 4

	fig, axs = plt.subplots(num_examples, num_cols, figsize=(23, 11))

	# gamma_predicted_G = chromatin_solver.compute_predicted_G(F_gamma_solution)

	# Select bins with the highest max values
	G_values, highest_G_indices = subset_select_highest_G_indices(G)

	from src.plot_helpers import create_sub_colormap

	reds = create_sub_colormap('Reds', 0.1, 0.9, 'Reds_smaller')
	blues = create_sub_colormap('Blues', 0.1, 0.9, 'Blues_smaller')

	highest_g = G_values.max()
	highest_f = F_kappa_curves[:, :, highest_G_indices].max()
	highest_val = max(highest_g, highest_f)

	ylims = -highest_val*0.05, highest_val*1.05

	from matplotlib.colors import ListedColormap
	from matplotlib.colors import LogNorm

	cmap = ListedColormap(plt.cm.plasma_r(np.linspace(0.2, 0.8, 256)))
	norm = LogNorm(vmin=kappa_values.min(), vmax=kappa_values.max())

	# # Plot each curve with the correct color from the colormap
	# for kappa, curve in zip(kappa_values, curve_data):
	#     color = cmap(norm(kappa))  # This handles the log transformation automatically
	#     plt.plot(x, curve, color=color)

	def plot_branch(ax, F_gamma_solution, indices, f_bin_index, color):
		ax.plot(F_gamma_solution[indices, f_bin_index].T, c=color,lw=3)
		ax.set_ylim(*ylims)

	for k_index in range(F_kappa_curves.shape[0]):

		kappa = kappa_values[k_index]
		F_gamma_solution = F_kappa_curves[k_index]

		for i in range(num_examples):

			ax_row = axs[i]

			if i != num_examples-1:
				for ax in ax_row:
					ax.set_xticks([])
					ax.set_yticks([])

			f_bin_index = highest_G_indices[i]

			if i == 0 and k_index == 0:
				ax_row[0].set_title("Data / Fit")
				ax_row[1].set_title("Initial branch")
				ax_row[2].set_title("Top branch")
				ax_row[3].set_title("Bottom branch")


			if k_index == 0:
				ax = ax_row[0]
				ax.plot(G[:, f_bin_index], c='black', lw=3, label="Raw data")
				#ax.plot(gamma_predicted_G[:, f_bin_index].T, c='red',
				#        lw=3, label="Optimal $\\gamma$ solution")
				ax.legend()
				ax.set_ylim(*ylims)

				# Add colorbar
				sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
				plt.colorbar(sm, ax=ax_row[1])

			color = cmap(norm(kappa))
			plot_branch(ax_row[1], F_gamma_solution, i_indices, f_bin_index, color)
			plot_branch(ax_row[2], F_gamma_solution, t_indices, f_bin_index, color)
			plot_branch(ax_row[3], F_gamma_solution, b_indices, f_bin_index, color)

	return fig
