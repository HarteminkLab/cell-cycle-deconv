
from math import comb
from matplotlib import pyplot as plt
from scipy.stats import norm
from src.helpers import calcH, createF, get_wavelet_kernel
from cc_src.sgd import get_gene_name_orf_name

import cvxpy as cp
import numpy as np
import pandas as pd


class Model:
	"""A model class to deconvolve gene expression data from CLOCCS cell cycle.

	Attributes:
		gene_name (str): Name of gene to deconvolve.
		gamma (float): Regularization parameter.
		orf_name (str): Name of ORF mapped to the specified gene.
		orf_id (int): Index of the specified ORF.
		g (list of float): Measured time series population data.
	"""
	
	def __init__(self, config, gene_or_orfname, gamma=None):

		self.orf_name, self.gene_name = get_gene_name_orf_name(gene_or_orfname)

		if gamma is None:
			gammas = config.xg_gammas
			gamma = gammas.loc[self.orf_name].gamma

		self.config = config
		self.gamma = gamma

		if self.config.wt1_df is not None:
			g1 = self.config.wt1_df.loc[self.orf_name].values
			self.g1 = g1
			self.g = g1

		self.initial_phase_map, self.top_phase_map, self.bottom_phase_map = config.intervals_wt1[-1]

		H1, self.Hpos = calcH(config.intervals_wt1, config.WT1_TIMEPOINTS)

		if self.config.has_two_replicates:
			g2 = self.config.wt2_df.loc[self.orf_name].values
			self.g2 = g2
			self.g = np.concatenate((g1, g2))

			H2, _ = calcH(config.intervals_wt2, config.WT2_TIMEPOINTS)
			self.H = np.concatenate((H1, H2))

		else:
			self.H = H1

	def get_f_it(self):
		f_it = []
		for phase in self.initial_phase_map.values():
			se = self.Hpos[phase[1]]
			f_it.extend([e for e in range(se[0], se[1])])
		for phase in self.top_phase_map.values():
			se = self.Hpos[phase[1]]
			f_it.extend([e for e in range(se[0], se[1])])
		f_it = np.array(f_it)
		return f_it

	def get_f_b(self):
		f_b = []
		for phase in self.bottom_phase_map.values():
			se = self.Hpos[phase[1]]
			f_b.extend([e for e in range(se[0], se[1])])
		f_b = np.array(f_b)
		return f_b

	def deconvolve(self):

		f_it = self.get_f_it()	
		f_b = self.get_f_b()

		# Mirroring
		f_b_mirror = np.concatenate((f_b, f_b))
		f_it_mirror = np.concatenate((f_it, np.flip(f_it)))
		factor_fb = 1.5

		W1 = get_wavelet_kernel(len(f_it_mirror))
		W2 = get_wavelet_kernel(len(f_b))
		W2pad = np.zeros((len(f_b), len(f_b)))
		W2 = np.concatenate((np.concatenate((W2, W2pad)), np.concatenate((W2pad, np.fliplr(W2)))), axis=1)

		# Convex optimization
		f = cp.Variable(self.H.shape[1])
		objective = cp.Minimize(cp.square(cp.pos(cp.norm(self.H@f/self.g - 1))) 
								+ self.gamma * (cp.norm(W1@f[f_it_mirror], 1) 
								+ factor_fb * cp.norm(W2@f[f_b_mirror], 1))/self.g.mean())
		constraints = [f >= 0]
		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=cp.CLARABEL)

		# predicted g
		self.pred_g = np.matmul(self.H, f.value)
		
		W1 = get_wavelet_kernel(len(f_it))
		W2 = get_wavelet_kernel(len(f_b))

		sn = (np.linalg.norm(np.matmul(W1, f.value[f_it]), 1) + np.linalg.norm(np.matmul(W2, f.value[f_b]), 1)) / np.mean(self.g)
		rn = np.square(np.clip(np.linalg.norm(np.matmul(self.H, f.value) / (self.g) - 1), 0, None))

		self.sn = sn
		self.rn = rn
		self.f = f 
		self.compute_ptr()


	def compute_ptr(self):
		"""Compute the peak to trough ratio"""
		from cc_src.peak_to_trough import compute_ptr
		self.cptr, self.dptr, self.ptr = compute_ptr(self, self.f.value)


	def plot_deconvolved_gene(self, title=None):

		# If I want to plot the initial branch,
		# I need the branch name: i
		# the phases:   R, CG1, postG1
		# and their associated timepoints and indices:
			# Indices is done
			# timepoints are looked up in the intervals object

		g1 = self.g1
		predicted_g = self.pred_g

		if self.config.has_two_replicates:
			g2 = self.g2
			g = np.concatenate([g1, g2])

			predicted_g1 = predicted_g[range(len(g1))]
			predicted_g2 = predicted_g[len(g1):]
		else:
			g = g1
			predicted_g1 = predicted_g

			g2 = None
			predicted_g2 = None

		f = self.f.value

		self.plot_deconvolved_f(f, g, g1, predicted_g1, g2, predicted_g2)


	def plot_deconvolved_f(self, f, g, g1, predicted_g1, g2=None, predicted_g2=None):

		fig, axs = plt.subplots(2, 4, figsize=(16, 7))
		(ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = np.array(axs).flatten()

		# -----------------

		timepoints1 = self.config.WT1_TIMEPOINTS
		ax0.plot(timepoints1, g1, color=self.color_for_key('raw'), lw=4)
		ax0.plot(timepoints1, predicted_g1, color=self.color_for_key('fit'), lw=4)
		ax0.set_yscale('log')

		# -----------------

		if g2 is not None:
			timepoints2 = self.config.WT2_TIMEPOINTS
			ax4.plot(timepoints2, g2, color=self.color_for_key('raw'), lw=4)
			ax4.plot(timepoints2, predicted_g2, color=self.color_for_key('fit'), lw=4)
			ax4.set_yscale('log')

		# -----------------

		def _plot_branch(ax, branch, ylim, start_offset=0, linestyle='solid'):
			"""Plot the branch coloring the individual phases within the branch"""
			phase_tp_idx_list = self.config.get_timepoints_phases_Hpositions_for_branch(branch)


			offset = 0
			if start_offset:
				offset = start_offset-phase_tp_idx_list[0][1].values[0]

			for phase, timepoints, indices in phase_tp_idx_list:
				ax.plot(timepoints+offset, f[indices], color=self.color_for_key(phase), 
					lw=5, linestyle=linestyle)

			ax.set_ylim(*ylim)

			# return timepoints in case we want to append more branches on to the plot
			return timepoints.values

		# ------------------

		for phase, indices in self.config.phase_columns.items():
			ax1.plot(indices, f[indices], color=self.color_for_key(phase), lw=5)

		ylim = 0, np.max(f)*1.1

		ax1.set_title("Deconvolved, f")

		_plot_branch(ax2, 't', ylim)
		ax2.set_title("Top branch")

		_plot_branch(ax3, 'b', ylim)
		ax3.set_title("Bottom branch")

		ax5.imshow(self.H, aspect='auto')
		ax5.set_title('Convolution kernel, H')

		_plot_branch(ax6, 'i', ylim)
		ax6.set_title("Initial branch")

		i_timepoints = _plot_branch(ax7, 'i', linestyle='dashed', ylim=ylim)
		#t_timepoints = _plot_branch(ax7, 't', start_offset=i_timepoints[-1], linestyle='dashed')
		_plot_branch(ax7, 'b', start_offset=i_timepoints[-1], linestyle='dashed', ylim=ylim)
		ax7.set_title("Single cell profile")

		#plt.suptitle(f"{self.gene_name} / {self.orf_name}, gamma={self.gamma:.3f}\nrn={self.rn:.2f}, sn={self.sn:.2f}\nptr={self.ptr:.2f}")


	def color_for_key(self, key):
		"""Predefined colors for phases and keys for gene plots"""

		return color_for_key(key)

	def save_deconvolved_outputs(self, index, out_dir):

		import sys

		orf_name = self.orf_name

		g_save_path = f'{out_dir}/{index}_g_{orf_name}.npy'
		f_save_path = f'{out_dir}/{index}_f_{orf_name}.npy'
		ptr_save_path = f'{out_dir}/{index}_ptr_{orf_name}.npy'
		meta_save_path = f'{out_dir}/{index}_meta_{orf_name}.csv'

		#---------- Save to disk -------------

		# Save the g to disk
		np.save(g_save_path, self.g)

		# Save the f to disk
		np.save(f_save_path, self.f)

		# Save meta information
		df = pd.DataFrame({'rn': self.rn, 'sn': self.sn, 'gm': self.gamma, 'ptr': self.ptr}, 
			index=[self.orf_name])
		df.to_csv(meta_save_path, float_format="%.4f")

		print(f"Saved to {g_save_path}...")
		print(f"Saved to {f_save_path}...")
		print(f"Saved to {meta_save_path}...")
		sys.stdout.flush()


def color_for_key(key):
	"""Predefined colors for phases and keys for gene plots"""

	color_map = {
		 "raw": np.array([158, 50, 50])/255.,
		 "fit": np.array([145, 180, 98])/255.,
		 "R": np.array([199, 148, 144])/255.,
		 "RG1": np.array([199, 148, 144])/255.,
		 "CG1": np.array([147, 168, 198])/255.,
		 "DG1": np.array([165, 197, 204])/255.,
		 "postG1": np.array([223, 192, 158])/255.,

		 "S": np.array([200, 192, 158])/255.,
		 "G2": np.array([223, 172, 158])/255.,

		 "H": np.array([100, 100, 100])/255.
	}

	return color_map[key]


def deconvolve_gene(config, gene_or_orfname, gamma=None, plot=False):
    model = Model(config=config, gene_or_orfname=gene_or_orfname, gamma=gamma)
    model.deconvolve()
    if plot: model.plot_deconvolved_gene()
    return model


def deconvolve_all_genes(config, save_dir):

	from src.timer import Timer

	# Initialize configuration and model
	timer = Timer()

	gene_orfs = config.all_orfs()

	# We will store all of the deconvolved f values into this matrix
	gene_fs_df = pd.DataFrame(np.zeros((len(gene_orfs), config.num_columns)))
	gene_fs_df.index = gene_orfs

	# Store ptr, rn, sn values into a meta data dataframe
	gene_meta_df = pd.DataFrame()
	gene_meta_df.index = gene_orfs

	print(f"Deconvolving {len(gene_orfs)} genes...")

	# Loop through all genes and store the deconvolved f values.
	index = 0
	for orf_name in gene_orfs:

		try:
			model = deconvolve_gene(config, orf_name)
		except IndexError:
			print(f"Error with orf: {orf_name}, index: {index}. Skipping...")
			index += 1
			continue

		gene_fs_df.loc[orf_name] = model.f.value

		(gene_meta_df.loc[orf_name, 'cptr'], gene_meta_df.loc[orf_name, 'dptr'], 
		gene_meta_df.loc[orf_name, 'ptr'], gene_meta_df.loc[orf_name, 'rn'], 
		gene_meta_df.loc[orf_name, 'sn']) = (model.cptr, model.dptr, model.ptr, model.rn, model.sn)

		if index % 100 == 0:
			print(f"   {index+1}/{len(gene_orfs)} - {timer.get_time()}")
			save_df(gene_fs_df, f'{save_dir}/deconvolved_fs.csv')
			save_df(gene_meta_df, f'{save_dir}/meta.csv')

		index += 1

	print(f"Completed. {timer.get_time()}")

	save_df(gene_fs_df, f'{save_dir}/deconvolved_fs.csv', silent=False)
	save_df(gene_meta_df, f'{save_dir}/meta.csv', silent=False)

	return gene_fs_df, gene_meta_df


def save_df(df, save_path, silent=True):
	df.to_csv(save_path)
	if not silent: print(f"Saved to {save_path}")

