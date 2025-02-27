
import numpy as np
import pandas as pd



# Expression find optimal kappa code
from src.expression_gamma_search import GeneExpressionFindOptimalGamma
from src.gene_expression import read_yl_vst_data_rep
from src.sgd import read_sgd_genes, get_orfname
from src.deconvolution_solver import DeconvolutionSolver
import matplotlib.pyplot as plt


RIBOSOMAL_GENES = ['RPL1A','RPL1B','RPL2A','RPL2B','RPL3','RPL4A','RPL4B',
   'RPL5','RPL16A','RPL16B','RPL17A','RPL17B','RPL18A','RPL18B']

DSE_GENES = ['DSE1', 'DSE2', 'DSE3', 'DSE4']

# HOUSEKEEPING_TFs = ['CBF1', 'ABF1', 'REB1']

CONTROL_GENES = ['CLB2', 'CLN2', 'MCM6', 'CDC45', 'SSK22'] + RIBOSOMAL_GENES[0:4]


def load_gene_expression(gene_name):

	gene_expression_data = read_yl_vst_data_rep(1)
	orf_name = get_orfname(gene_name)
	gene_expression = gene_expression_data.loc[orf_name]
	gene_expressions_tpm = pd.read_csv(
		'datasets/yl_cell_cycle/replicate1_gene_expression_TPM.csv')
	gene_expressions_tpm = gene_expressions_tpm.set_index('orf_name')


	gene_expression_tpm = gene_expressions_tpm.loc[orf_name]
	g = np.log2(gene_expression_tpm.values+1)
	return g



class FindKappaExpression(object):
	"""Find the optimal kappa that balances daughter-specific expression
	signal and controls for false-positive signal of non-daughter-specific
	genes during DG1"""

	def __init__(self, config):
		self.config = config

	def compute_phase_sums_kappa(self, gene_name, kappa):
		"""Compute the CG1, DG1, postG1 masses for a given gene and kappa value 
		regularization of (CG1/DG1)"""

		config = self.config

		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		cg1_indices = config.get_Hpositions_for_phase('CG1')
		dg1_indices = config.get_Hpositions_for_phase('DG1')
		postg1_indices = config.get_Hpositions_for_phase('postG1')

		g = load_gene_expression(gene_name)
		expression_find_gamma = GeneExpressionFindOptimalGamma(config=config, gene_expression=g)
		expression_find_gamma.find_optimal_gamma(plot=False, verbose=False, kappa=kappa)

		optimal_f = expression_find_gamma.retrieve_solution()

		def compute_phase_sums(optimal_f):
			return optimal_f[cg1_indices].sum(), \
				optimal_f[dg1_indices].sum(), optimal_f[postg1_indices].sum()

		phase_sums = compute_phase_sums(optimal_f)
		
		return expression_find_gamma, optimal_f, phase_sums

	def sweep_kappas(self):

		from src.timer import Timer
		k_min, k_max = 1e-6, 1e-2
		num_kappas = 21

		logks = np.linspace(np.log10(k_min), np.log10(k_max), num_kappas)
		kappas = 10**logks
		# kappas = np.linspace(k_min, k_max, 20)

		timer = Timer()
		gene_sums_df = pd.DataFrame()

		gene_names = DSE_GENES+CONTROL_GENES

		gene_f_solutions = {}

		for i, kappa in enumerate(kappas):
			print(f"[{i+1}/{len(kappas)}]Kappa=", kappa)
			print("\t", end="")

			for gene_name in gene_names:
				print(gene_name, end=",")
				expression_find_gamma, optimal_f, phase_sums = \
					self.compute_phase_sums_kappa(gene_name, kappa=kappa)
				gene_sums_row = pd.DataFrame(phase_sums, 
											 index=['CG1_sum', 'DG1_sum', 'postG1_sum'], 
											 columns=[gene_name]).T
				gene_sums_row['kappa'] = kappa
				gene_sums_df = pd.concat([gene_sums_df, gene_sums_row])
				
				gene_f_solutions[(gene_name, kappa)] = optimal_f
				
			print("\t", end="")
			timer.print_time()
			print()

		self.gene_f_solutions = gene_f_solutions
		self.gene_sums_df = gene_sums_df


	def compute_l2tb(self):
		from src.peak_to_trough import compute_quantile_ptr
		from scipy.stats import pearsonr

		dse_genes = DSE_GENES
		control_genes = CONTROL_GENES#RIBOSOMAL_GENES[0:4]

		config = self.config

		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		cg1_indices = config.get_Hpositions_for_phase('CG1')
		dg1_indices = config.get_Hpositions_for_phase('DG1')

		gene_f_solutions = self.gene_f_solutions

		df = pd.DataFrame(gene_f_solutions).T
		df = df.reset_index().rename(columns={'level_0': 'gene', 'level_1': 'kappa'})\
		    .set_index(['gene', 'kappa'])
		self.gene_solutions_df = df
		summary_df = df.copy()[[]]

		# Compute the l2 norm of the top and bottom branches for each gene
		for index, row in df.iterrows():

		    l2 = np.linalg.norm(row[b_indices].values - row[t_indices].values) / len(t_indices)
		    ptr = np.quantile(row[dg1_indices].values, q=0.9) / np.quantile(row[cg1_indices].values, q=0.1)

		    summary_df.loc[index, 'l2_tb'] = l2
		    summary_df.loc[index, 'ptr'] = ptr

		# The means will be used for the signal to noise computation
		avg_dse_ratio = summary_df.loc[dse_genes].groupby('kappa').mean()
		avg_control_ratio = summary_df.loc[control_genes].groupby('kappa').mean()

		# Normalize to std 1
		ratio_std = 1#np.concatenate([avg_dse_ratio, avg_control_ratio]).std()

		self.avg_dse_ratio = avg_dse_ratio/ratio_std
		self.avg_control_ratio = avg_control_ratio/ratio_std

		self.l2_tb_eps = 1
		self.snr_l2 = ((self.avg_dse_ratio.l2_tb/ratio_std+self.l2_tb_eps)/
							(self.avg_control_ratio.l2_tb/ratio_std+self.l2_tb_eps))

	def plot_l2_tb(self):

		plt.figure(figsize=(6, 3))
		plt.subplot(1, 2, 1)
		plt.plot(self.avg_dse_ratio.l2_tb, label="Daughter-specific genes")
		plt.plot(self.avg_control_ratio.l2_tb, label="Control genes")

		plt.title("L2 norm comparison,\ndaughter vs control genes")
		plt.ylabel("L2 norm of top/bottom branches")
		plt.xlabel("Kappa, regularization")
		plt.legend()
		plt.xscale('log')

		plt.subplot(1, 2, 2)
		xs = self.avg_dse_ratio.index.values
		plt.plot(xs, self.snr_l2, label="Ratio (D/C)")
		plt.scatter(xs, self.snr_l2)
		plt.xlabel("Kappa, regularization")
		plt.title(f"Ratio of daughter vs control L2\n(pseudo-count={self.l2_tb_eps})")
		plt.legend()
		plt.xscale('log')


	def plot_top_bottom_curves(self, kappa_idx):

		from src.expression_chromatin_plots import draw_phase_label_annotations

		dse_genes = DSE_GENES
		control_genes = CONTROL_GENES #RIBOSOMAL_GENES[:4]

		dat_df = self.gene_solutions_df.reset_index().set_index(['kappa', 'gene'])
		kappa = dat_df.index.levels[0][kappa_idx]

		# dat_df.loc[:] = dat_df.values - dat_df.mean(axis=1).values[:, None]

		config = self.config
		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		fig = plt.figure(figsize=(4, 13))

		num_examples = 12
		plt.subplot(1+num_examples, 2, 1)

		plt.plot(t_tps, dat_df.loc[kappa, t_indices].loc[dse_genes].T, c='red', alpha=0.15)
		plt.plot(t_tps, dat_df.loc[kappa, b_indices].loc[dse_genes].T, c='blue', alpha=0.15)
		plt.title("Daughter-specific genes")
		plt.xticks([])

		for i in range(len(dse_genes)):
			plt.subplot(1+num_examples, 2, 3+i*2)
			plt.plot(t_tps, dat_df.loc[kappa, t_indices].loc[dse_genes].iloc[i], c='red')
			plt.plot(t_tps, dat_df.loc[kappa, b_indices].loc[dse_genes].iloc[i], c='blue')
			plt.xticks([])
			plt.xlim(t_tps[0], t_tps[-1])
			plt.ylim(-0.1, 15)
			plt.title(dse_genes[i])

		for i in range(len(control_genes)):
			plt.subplot(1+num_examples, 2, 4+i*2)
			plt.plot(t_tps, dat_df.loc[kappa, t_indices].loc[control_genes].iloc[i], c='red')
			plt.plot(t_tps, dat_df.loc[kappa, b_indices].loc[control_genes].iloc[i], c='blue')
			plt.yticks([])
			plt.xlim(t_tps[0], t_tps[-1])
			plt.xticks([])
			plt.ylim(-0.1, 15)
			plt.title(control_genes[i])
