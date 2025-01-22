
from math import comb
from matplotlib import pyplot as plt
from scipy.stats import norm
from src.helpers import calcH

from src.calcH_single_g1 import calcH as calcH_single_g1
from src.single_G1_config import Config as Config_single_G1
from src.sgd import get_gene_name_orf_name
import numpy as np
import pandas as pd
from src.utils import print_fl


class Model:
	"""A model class to deconvolve gene expression data from CLOCCS cell cycle.

	Attributes:
		gene_name (str): Name of gene to deconvolve.
		gamma (float): Regularization parameter.
		orf_name (str): Name of ORF mapped to the specified gene.
		orf_id (int): Index of the specified ORF.
		g (list of float): Measured time series population data.
	"""
	
	def __init__(self, config, gene_or_orfname, gamma=0.0, for_chromatin_deconv=False,
		expression_data=None):

		if gene_or_orfname is None:
			self.orf_name = None
			self.gene_name = None
		else:
			self.orf_name, self.gene_name = get_gene_name_orf_name(gene_or_orfname)

		self.config = config
		self.gamma = gamma

		if not for_chromatin_deconv:
			g1 = expression_data.loc[self.orf_name].values
			self.g1 = g1
			self.g = g1

		H1 = config.H

		# todo: testing refactor of config for single replicate
		if False:

			if not for_chromatin_deconv:
				# Load g2
				g2 = self.config.wt2_df.loc[self.orf_name].values
				self.g2 = g2

				# Copy number correction for g1 and g2
				if config.copy_correction is not None:
					print_fl("Applying copy number correction")
					copy_correction_vector1 = config.copy_correction[0].loc[self.orf_name]
					copy_correction_vector2 = config.copy_correction[1].loc[self.orf_name]
					self.g1 = self.g1 * copy_correction_vector1
					self.g2 = self.g2 * copy_correction_vector2

				self.g = np.concatenate((g1, g2))

			H2, _ = calcH_function(config.intervals_wt2, config.timepoints)
			self.H = np.concatenate((H1, H2))

		else:
			self.H = H1

	def deconvolve_find_optimal_gamma(self, silence=True):
		from src.find_gamma import FindOptimalGamma
		find_gamma = FindOptimalGamma(self)
		find_gamma.find_optimal(silence=silence)


	def deconvolve(self, enforce_non_negative=True):

		from src.deconvolution_solver import DeconvolutionSolver

		deconvolution_solver = DeconvolutionSolver(self.config, self.g, self.H, self.gamma, 
			padding_type='both')
		self.f, self.sn, self.rn = deconvolution_solver.deconvolve()

		# predicted g
		self.pred_g = np.matmul(self.H, self.f)
		self.compute_ptr()


	def compute_ptr(self):
		"""Compute the peak to trough ratio"""
		from src.peak_to_trough import compute_ptr
		self.cptr, self.dptr, self.ptr = 0, 0, 0#compute_ptr(self.config, self.f)


	def plot_deconvolved_gene(self, title=None, abbreviated=False):

		# If I want to plot the initial branch,
		# I need the branch name: i
		# the phases:   R, CG1, postG1
		# and their associated timepoints and indices:
			# Indices is done
			# timepoints are looked up in the intervals object

		g1 = self.g1
		predicted_g = self.pred_g

		if False: #self.config.has_two_replicates:
			g2 = self.g2
			g = np.concatenate([g1, g2])

			predicted_g1 = predicted_g[range(len(g1))]
			predicted_g2 = predicted_g[len(g1):]
		else:
			g = g1
			predicted_g1 = predicted_g

			g2 = None
			predicted_g2 = None

		f = self.f

		fig = self.plot_deconvolved_f(f, g, g1, predicted_g1, g2, predicted_g2, abbreviated=abbreviated)
		return fig


	def plot_H(self, H=None):

		if H is None:
			H = self.H

		rg1_cols = self.config.get_Hpositions_for_phase('RG1')
		cg1_cols = self.config.get_Hpositions_for_phase('CG1')
		dg1_cols = self.config.get_Hpositions_for_phase('DG1')

		H_cols = np.array([H.shape[1]-1])

		s_cols = self.config.get_Hpositions_for_phase('S')
		g2m_cols = self.config.get_Hpositions_for_phase('G2M')
		phases = ['H', 'RG1', 'CG1', 'DG1', 'S', 'G2M']
		cols_list = [H_cols, rg1_cols, cg1_cols, dg1_cols, s_cols, g2m_cols]

		plt.figure(figsize=(16, 3))
		plt.subplot(1, 2, 1)

		plt.imshow(H, vmax=H[:, :-1].max(), aspect='auto', cmap='Reds',
				  extent=[0, H.shape[1], self.config.timepoints[-1], 0])

		plt.subplot(1, 2, 2)

		x = self.config.timepoints

		prev = np.zeros(len(x))
			
		for i in range(len(phases)):

			phase = phases[i]
			cols = cols_list[i]

			color = color_for_key(phase)

			y = prev+H[:, cols].sum(axis=1)

			plt.fill_between(x, prev, y, color=color, label=phase)
			
			prev = y
		plt.legend()


	def plot_deconvolved_f(self, f, g, g1, predicted_g1, g2=None, predicted_g2=None, abbreviated=False):

		if abbreviated:
			fig, axs = plt.subplots(2, 2, figsize=(9, 9))
			plt.subplots_adjust(top=0.77)
			(ax0, ax1, ax4, ax5) = np.array(axs).flatten()
		else:
			fig, axs = plt.subplots(2, 4, figsize=(16, 9))
			plt.subplots_adjust(top=0.77)

			(ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = np.array(axs).flatten()
			

		# -----------------

		timepoints1 = self.config.timepoints
		timepoints2 = self.config.timepoints

		if timepoints2 is not None:
			xlims = timepoints1[0], max(timepoints1[-1], timepoints2[-1])
		else:
			xlims = timepoints1[0], timepoints1[-1]
			ax4.spines['top'].set_visible(False)
			ax4.spines['bottom'].set_visible(False)
			ax4.spines['left'].set_visible(False)
			ax4.spines['right'].set_visible(False)
			ax4.set_xticks([])
			ax4.set_yticks([])

		ax0.plot(timepoints1, g1, color=self.color_for_key('raw'), lw=4)
		ax0.plot(timepoints1, predicted_g1, color=self.color_for_key('fit'), lw=4)
		ax0.set_xticks(np.arange(0, xlims[1]+50, 50))
		ax0.set_xticks(np.arange(0, xlims[1]+10, 10), minor=True)
		ax0.set_xlim(*xlims)

		# -----------------

		if g2 is not None:
			timepoints2 = self.config.timepoints
			ax4.plot(timepoints2, g2, color=self.color_for_key('raw'), lw=4)
			ax4.plot(timepoints2, predicted_g2, color=self.color_for_key('fit'), lw=4)
			# ax4.set_yscale('log')

		# -----------------

		def _plot_branch(ax, branch, ylim, start_offset=0, linestyle='solid'):
			"""Plot the branch coloring the individual phases within the branch"""
			phase_tp_idx_list = self.config.branch_Hpos_df.loc[branch]

			offset = 0
			#if start_offset:
				#offset = start_offset-phase_tp_idx_list[0][1].values[0]

			phases = phase_tp_idx_list.index.values
			for phase in phases:

				branch = 'i'
				branch_tps = phase_tp_idx_list.loc[phase].timepoint_start
				indices = phase_tp_idx_list.loc[phase].Hpos

				ax.plot(branch_tps+offset, f[indices], color=self.color_for_key(phase), 
					lw=5, linestyle=linestyle)

			ax.set_ylim(ylim[0], ylim[1])

			return branch_tps.values + offset

		# ------------------

		# Set the ylim appropriate to the values in f
		f_values_for_lim = f[:-1]
		diff = np.max(f_values_for_lim) - np.min(f_values_for_lim)
		ylim = np.min(f_values_for_lim)-diff*0.1, np.min(f_values_for_lim)+diff*1.1

		# Halted cells are the last element in f
		halted_f = f[len(f)-1]
		ax1.scatter(-1, halted_f, color='gray', s=50, marker='H')

		phases = ['RG1', 'CG1', 'DG1', 'postG1']
		for phase in phases:
			indices = self.config.get_Hpositions_for_phase(phase)
			plot_f_values = f[indices]
			ax1.plot(indices, plot_f_values, color=self.color_for_key(phase), lw=5)
		ax1.set_ylim(*ylim)

		ax1.set_title("Deconvolved, f")

		if not abbreviated:
			_plot_branch(ax3, 't', ylim)
			ax3.set_title("Top branch")

		if not abbreviated:
			_plot_branch(ax2, 'i', ylim)
			ax2.set_title("Initial branch")

		if not abbreviated:
			i_timepoints = _plot_branch(ax5, 'i', linestyle='solid', ylim=ylim)
			t_timepoints = _plot_branch(ax5, 't', linestyle='solid', ylim=ylim, 
				start_offset=i_timepoints[-1])
			_plot_branch(ax5, 'b', start_offset=t_timepoints[-1], linestyle='solid', ylim=ylim)
			ax5.set_title("Single cell profile")

		if not abbreviated:
			_plot_branch(ax7, 'b', ylim)
			ax7.set_title("Bottom branch")

		im = ax6.imshow(self.H, aspect='auto', interpolation='Nearest', cmap='Spectral_r', vmax=0.1)
		ax6.spines['top'].set_visible(False)
		ax6.spines['bottom'].set_visible(False)
		ax6.spines['left'].set_visible(False)
		ax6.spines['right'].set_visible(False)
		ax6.set_xticks([])
		ax6.set_yticks([])
		ax6.set_title('Convolution kernel, H')

		title = f"{self.gene_name} / {self.orf_name}, gamma={self.gamma:.4f}\nrn={self.rn:.4f}, sn={self.sn:.2f}"

		#if self.config.name is not None:
		#	title = f"{self.config.name}\n" + title

		plt.suptitle(title, fontsize=23)
		return fig


	def color_for_key(self, key):
		"""Predefined colors for phases and keys for gene plots"""

		return color_for_key(key)

	def save_deconvolved_outputs(self, out_dir, index):

		import sys

		orf_name = self.orf_name

		g_save_path = f'{out_dir}/{index}_g_{orf_name}.npy'
		f_save_path = f'{out_dir}/{index}_f_{orf_name}.npy'
		ptr_save_path = f'{out_dir}/{index}_ptr_{orf_name}.npy'
		meta_save_path = f'{out_dir}/{index}_meta_{orf_name}.csv'
		plot_save_path = f'{out_dir}/{index}_plot_{orf_name}.png'

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
		print(f"Saved to {plot_save_path}...")
		sys.stdout.flush()


def color_for_key(key):
	"""Predefined colors for phases and keys for gene plots"""

	color_map = {
		 "raw": np.array([158, 50, 50])/255.,
		 "fit": np.array([145, 180, 98])/255.,
		 "R": np.array([199, 148, 144])/255.,
		 "RG1": np.array([199, 148, 144])/255.,

		 "CG1": np.array([214, 170, 129])/255.,
		 "DG1": np.array([227, 194, 163])/255.,

		 "Delta": np.array([227, 194, 163])/255.,
		 "postG1": np.array([223, 192, 158])/255.,
		 "RpostG1": np.array([200, 170, 140])/255.,

		 "G2M": np.array([147, 168, 198])/255.,
		 "S": np.array([158, 189, 140])/255.,

		 "H": np.array([100, 100, 100])/255.
	}

	color_map['G2/M'] = color_map['G2M']

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

		gene_fs_df.loc[orf_name] = model.f

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


def create_mirror(ind_vec):

	# Determine if the input vector is a power of 2,
	# if it is not, use an inset that: after mirroring
	# the output vector will be a power of 2
	from src.helpers import compute_closest_pow2
	vec_len = len(ind_vec)
	closest_pow2 = compute_closest_pow2(vec_len)

	if closest_pow2-vec_len > 0:
		inset_index = (closest_pow2 - vec_len)//2

	# Otherwise use half of the input vector
	else:
		inset_index = len(ind_vec) // 2

	ind_vec = np.concatenate([np.flip(ind_vec[:inset_index]), 
		ind_vec, np.flip(ind_vec[-inset_index:])])

	return ind_vec
