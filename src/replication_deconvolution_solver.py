
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel, pad_with_subset
from src.utils import print_fl
from src.geneset import get_deconvolved_geneset


from src.wavelets_2d_linalg import decompose_flattened_kron_coeffs, \
	create_kron_wavelet2d_convolution_matrices


class ReplicationChromatinDeconvolveSolver:
	"""
	In this class, we will perform some changes to the chromatin deconvolution 
	to handle the copy number transition from 1 to 2. We expect this transition to be a 
	step-wise transition so we will use Haar. And the smoothing constraints will be 
	different as we expect from postG1 to G1, the contraint no longer needs to be smooth.
	"""

	def __init__(self, mnase_analysis_rep1, mnase_analysis_rep2, solver=cvxpy.MOSEK, wavelet="Haar"):

		self.solver = solver
		self.wavelet = wavelet
		self.geneset = get_deconvolved_geneset()
		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.mnase_analysis_rep2 = mnase_analysis_rep2

	def plot_raw_data(self):
		plt.figure(figsize=(4, 2))
		plt.plot(self.G[:, 0])

	def define_deconvolution_problem(self):

		solver = self.solver
		config = self.config
		G = self.G
		H = self.H
		
		# We will add a very small value to g, to avoid divide by zero errors
		eps = 1e-5
		G = G + eps

		f_i = config.get_Hpositions_for_branch('i')
		f_t = config.get_Hpositions_for_branch('t')
		f_b = config.get_Hpositions_for_branch('b')

		f_dg1 = config.get_Hpositions_for_phase('DG1')
		f_rg1 = config.get_Hpositions_for_phase('RG1')
		f_cg1 = config.get_Hpositions_for_phase('CG1')
		f_pg1 = config.get_Hpositions_for_phase('postG1')

		# Smooth each branch separately, but not the end of postG1 into G1.
		# As we now expect the end of G2M to be copy number 2, and the start
		# of G1 to be copy number 1.

		# We can try mirroring to handle edge effects though.
		f_i_mirror = create_mirror(f_i)
		f_t_mirror = create_mirror(f_t)
		f_b_mirror = create_mirror(f_b)

		W1 = get_wavelet_kernel(len(f_i_mirror), type=self.wavelet)
		W2 = get_wavelet_kernel(len(f_t_mirror), type=self.wavelet)
		W3 = get_wavelet_kernel(len(f_b_mirror), type=self.wavelet)

		g_mean = G.mean()

		# -------- Define the optimization ------------

		# f whose rows span the columns of H
		# and columns are the length of g's columns
		f = cvxpy.Variable((H.shape[1], G.shape[1]))

		self.gamma = cvxpy.Parameter(nonneg=True, name='gamma')

		# with the updated alpha, the i t and b are approximately all the same length
		self.factor_i = 1.

		smooth_f_i_result = W1@f[f_i_mirror]
		smooth_f_t_result = W2@f[f_t_mirror]
		smooth_f_b_result = W3@f[f_b_mirror]

		# -------------------------------------------------------------------------

		elementwise_result = cvxpy.multiply(H@f, 1.0/G) - 1

		n = G.shape[0]
		m = G.shape[1]
		u = H.shape[1]

		constraints = [f >= 0]

		# ------ Constrain monotonic transitions ----------

		# End of G1 to postG1
		constraints.append(f[f_pg1[0]] >= f[f_cg1[-1]])
		constraints.append(f[f_pg1[0]] >= f[f_dg1[-1]])
		constraints.append(f[f_pg1[0]] >= f[f_rg1[-1]])
		
		# RG1
		for i in range(1, len(f_rg1)):
			index = f_rg1[i]
			prev_index = f_rg1[i-1]
			constraints.append(f[index] >= f[prev_index])

		# DG1
		for i in range(1, len(f_dg1)):
			index = f_dg1[i]
			prev_index = f_dg1[i-1]
			constraints.append(f[index] >= f[prev_index])

		# CG1
		for i in range(1, len(f_cg1)):
			index = f_cg1[i]
			prev_index = f_cg1[i-1]
			constraints.append(f[index] >= f[prev_index])

		# Post G1
		for i in range(1, len(f_pg1)):
			index = f_pg1[i]
			prev_index = f_pg1[i-1]
			constraints.append(f[index] >= f[prev_index])

		# Halted cells are equal to Recovery cells
		constraints.append(f[index] == f[-1])

		# -------------------------------------------------

		objective = cvxpy.Minimize(

			cvxpy.sum(cvxpy.norm(elementwise_result, 'fro')**2) +

			self.gamma * (self.factor_i*cvxpy.sum(cvxpy.abs(smooth_f_i_result)) +
							cvxpy.sum(cvxpy.abs(smooth_f_t_result)) + 
							cvxpy.sum(cvxpy.abs(smooth_f_b_result)) 
							)/g_mean  
		)

		# -------- End definition ------------

		# Perform the convex optimization
		self.prob = cvxpy.Problem(objective, constraints)
		self.f = f

	def plot_replication_timing(self):

		from src.sgd import get_chromosome_length

		chr_genes = self.chr_genes
		chrom = self.chrom

		config = self.config
		chrom_len = get_chromosome_length(chrom)

		self.compute_replication_timing()

		plt.figure(figsize=(13, 2))
		plt.plot(chr_genes.start, self.repl_timing_df.timing, c='black', lw=0.5, ls='dotted')
		plt.scatter(chr_genes.start, self.repl_timing_df.timing, s=2, c='black')
		plt.ylim(60, 20)
		plt.xlim(0, chrom_len)
		plt.title(f"Combined Haar model replicate profile, chr{chrom}")

	def compute_replication_timing(self, threshold = 0.75):
		"""Compute the replication timing for all genes, assume that we can just
		use the mother timing for now, as we expect replication to occur in S-phase
		and each branch shares the same PostG1"""

		config = self.config
		t_pos = config.get_Hpositions_for_branch('t')
		t_tps = config.get_timepoints_for_branch('t')

		lambda_val = config.intervals_wt1[0][1]
		gamma1 = config.intervals_wt1[0][7]
		gamma2 = config.intervals_wt1[0][8]
		alpha = config.intervals_wt1[0][5]

		cg1_len = alpha + lambda_val*gamma1

		time_indices = np.argmax((self.f[t_pos] > 1+threshold), axis=0)
		repl_timing = t_tps[time_indices] + cg1_len

		repl_timing_df = self.chr_genes[[]].copy()
		repl_timing_df['timing'] = repl_timing

		self.repl_timing_df = repl_timing_df

	def plot_replication_hm(self, normalize=False, mask=False):
		f = self.f.copy()

		if normalize:
			f_norm = f
			f_norm = f_norm / f_norm.max(axis=0).reshape((1, -1))
			f = f_norm+1.

		config = self.config
		
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		plt.figure(figsize=(13, 3))

		if mask:
			f = (f > 1.75) + 1.
		
		def plot_repl_im(f):
			plt.imshow(f, origin='lower', aspect='auto', vmin=1, vmax=2)
		
		plt.subplot(3, 1, 1)
		plot_repl_im(f[i_indices])
		
		plt.subplot(3, 1, 2)
		plot_repl_im(f[t_indices])
		
		plt.subplot(3, 1, 3)
		plot_repl_im(f[b_indices])

		plt.suptitle(f"Combined model, chr{self.chrom}, gamma={self.gamma.value}")


	def plot_raw_predicted(self):
		pred_G = self.H @ self.f

		def plot_repl_im(f):
			plt.imshow(f, origin='lower', aspect='auto', vmin=1, vmax=2.)

		plt.figure(figsize=(13, 3))
		plt.subplot(2, 1, 1)
		plot_repl_im(self.G)

		plt.subplot(2, 1, 2)
		plot_repl_im(pred_G)


	def solve(self, gamma_value, verbose=False):

		self.gamma.value = gamma_value
		self.verbose = verbose

		# The epsilon value affects the precision of the solver
		self.result = self.prob.solve(solver=self.solver, warm_start=True, 
			verbose=self.verbose, eps=1e-4)
		f = self.f.value

		self.f = f

		return f

	def plot_result(self, i):
		plt.figure(figsize=(13, 2))
		config = self.config

		f = self.f[:, i]

		f_dg1 = config.get_Hpositions_for_phase('DG1')
		f_rg1 = config.get_Hpositions_for_phase('RG1')
		f_cg1 = config.get_Hpositions_for_phase('CG1')
		f_pg1 = config.get_Hpositions_for_phase('postG1')

		f_rg1_tps = config.get_phase_timepoints_for_phase('RG1')
		f_cg1_tps = config.get_phase_timepoints_for_phase('CG1')
		f_dg1_tps = config.get_phase_timepoints_for_phase('DG1')
		f_pg1_tps = config.get_phase_timepoints_for_phase('postG1')

		plt.subplot(1, 4, 1)
		plt.plot(f_rg1_tps, f[f_rg1])
		plt.plot(f_pg1_tps, f[f_pg1])
		plt.title("Recovery")

		plt.subplot(1, 4, 2)
		plt.plot(f_cg1_tps, f[f_cg1])
		plt.plot(f_pg1_tps, f[f_pg1])
		plt.title("Mother")

		plt.subplot(1, 4, 3)
		plt.plot(f_dg1_tps, f[f_dg1])
		plt.plot(f_pg1_tps, f[f_pg1])
		plt.title("Daughter")

		plt.subplot(1, 4, 4)
		plt.plot(self.G[:, i])
		plt.plot(self.H@f)
		plt.title("Predicted/Raw")

	def set_chromosome(self, chrom):
		self.chrom = chrom
		self.chr_genes = self.geneset[self.geneset.chr == chrom]

		g1 = self.get_g_for_chrom(self.mnase_analysis_rep1, self.chrom)
		g2 = self.get_g_for_chrom(self.mnase_analysis_rep2, self.chrom)
		self.G = np.concatenate([g1, g2])

	def load_combined_config(self):

		from src.config import load_yl_rg1_vst_config
		from src.helpers import calcH
		from src.global_config import GlobalConstants

		config1 = load_yl_rg1_vst_config(1)
		config2 = load_yl_rg1_vst_config(2)
		H1, Hpos = calcH(config1.intervals_wt1, GlobalConstants.CHROM_WT1_TIMEPOINTS)
		H2, Hpos = calcH(config2.intervals_wt1, GlobalConstants.CHROM_WT2_TIMEPOINTS)

		# Use config1 for timepoints
		self.config = config1

		self.H = np.concatenate([H1, H2])

	def get_g_for_chrom(self, mnase_analysis, chrom):
	    """Get the g curve for deconvolution for a chromosome"""
	    chr_curves = mnase_analysis.normalized_bin_curves.join(self.chr_genes[[]], how='inner')
	    g = chr_curves.values.T

	    # Add 1 to enforce copy number values 1-2, also easier to deconvolve values
	    # away from 0.
	    g = g + 1.
	    return g


	def save_results(self, directory):

		f_file = f"{directory}/chr{self.chrom}_f.npy"
		repl_file = f"{directory}/chr{self.chrom}_repl.csv"

		np.save(f_file, self.f)
		self.repl_timing_df.to_csv(repl_file)

		print(f"Saved {f_file}")
		print(f"Saved {repl_file}")


def create_mirror(ind_vec):
	ind_vec_n_2 = len(ind_vec) // 2
	ind_vec_mirror = np.concatenate([np.flip(ind_vec[:ind_vec_n_2]), ind_vec, 
		np.flip(ind_vec[-ind_vec_n_2:])])
	return ind_vec_mirror

