


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import load_yl_replicate1_rg1_alpha_vst_config
from src.dynamic_config_alpha import create_dynamic_alpha_config
from src.model import Model
from src.chromatin_model import ChromatinModel


class DeconvReplicationProfileAnalysis:
	"""Class to handle deconvolution of mnase seq occupancy 10k bins for 
	computing the replication timing

	Goals:
	1. Identify optimal alpha value from chr4 profiles for replicate 1 and 2
	2. Compute the profiles for all chromosomes and save to disk
	"""

	def __init__(self, mnase_analysis_rep1, mnase_analysis_rep2):
		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.mnase_analysis_rep2 = mnase_analysis_rep2

	def compare_early_late_deconv_bin_curves(self, alpha, replicate):
		"""Show the deconvolution of two example bin curves, early and late"""

		self.replicate = replicate

		if replicate == 1:
			mnase_analysis = self.mnase_analysis_rep1
		else:
			mnase_analysis = self.mnase_analysis_rep2

		# An early and late replicating gene
		from src.sgd import get_orfname
		orf_early = get_orfname('VPS8')
		orf_late = get_orfname('SSK22')

		early_bin_curve = mnase_analysis.normalized_bin_curves.loc[orf_early]
		late_bin_curve = mnase_analysis.normalized_bin_curves.loc[orf_late]

		dynamic_config = create_dynamic_alpha_config(alpha, replicate)

		gene_name = "CLN1"
		model = Model(dynamic_config, gene_name)

		model.g = early_bin_curve+1.
		model.g1 = early_bin_curve+1.
		model.deconvolve_find_optimal_gamma()
		_ = model.plot_deconvolved_gene()
		plt.suptitle(f"Early replicating 10k bin, $\\alpha$={alpha}\nsn={model.sn:.2f}, rn={model.rn:.2f}"
			f", gamma={model.gamma:.3f}, Replicate 1", fontsize=32)

		model.g = late_bin_curve+1.
		model.g1 = late_bin_curve+1.
		model.deconvolve_find_optimal_gamma()

		_ = model.plot_deconvolved_gene()
		plt.suptitle(f"Late replicating 10k bin, $\\alpha$={alpha}\nsn={model.sn:.2f}, rn={model.rn:.2f}"
			f"\ngamma={model.gamma:.3f}, "
			f"Replicate {self.replicate}", fontsize=32)

	def create_chr_bin_curves(self, replicate=1, chrom=4):
		"""Select the chromosome for the bin curves we will deconvolve"""

		print(f"Setting up bin curves for replicate {replicate}, chromosome {chrom}")

		if replicate == 1:
			mnase_analysis = self.mnase_analysis_rep1
		else:
			mnase_analysis = self.mnase_analysis_rep2

		from src.geneset import get_deconvolved_geneset
		geneset = get_deconvolved_geneset()
		self.bin_curves_G_df = mnase_analysis.normalized_bin_curves.dropna()
		self.chr_bin_curves = self.bin_curves_G_df.join(geneset[geneset['chr'] == chrom][[]], how='inner')

		# Deconvolve the normalized bin curves, will this allow for us 
		# to create a more high resolution replication timing profile?
		print("Setting up model with temporary gene. (Actual deconvolution will use chrom bin curves, not gene data)...")
		self.config = create_dynamic_alpha_config(0, self.replicate)
		self.chrom_model = ChromatinModel(self.config)
		self.chrom_model.load_mnase_gene("CLN1") # todo: dummy gene, we will not actually be deconvolving gene


	def get_f_for_alpha(self, alpha):
		"""Deconvolve the bin curves for a given alpha"""

		config = create_dynamic_alpha_config(alpha, self.replicate)
		self.config = config
		
		# Set the config to the updated alpha
		chrom_model = self.chrom_model
		chrom_model.config = config
		chrom_model.create_deconvolution_bins()
		
		# Set G to the 10k occupancy bin curves for the chromosome
		chrom_model.G = self.chr_bin_curves.T.values+1.
		
		chrom_model.setup_deconv_model()
		chrom_model.gamma = 0.001
		chrom_model.deconvolve()

		f = chrom_model.deconvolved_f()

		return f

	def alpha_search(self, alphas = np.arange(0, 30, 1)):
		"""Deconvolve the set of alpha values"""

		# Setup to get H dimensions
		chrom_model = self.chrom_model
		chrom_model.setup_deconv_model()

		# Dimensions of the returned alpha search f images
		u, m = self.chr_bin_curves.shape[0], chrom_model.deconv_model.H.shape[1]
		
		alpha_fs = np.zeros((len(alphas), m, u))

		from src.timer import Timer

		timer = Timer()

		for i in range(len(alphas)):
			alpha = alphas[i]
			print(f"{i}/{len(alphas)}, alpha={alpha}")
			deconvolved_f = self.get_f_for_alpha(alpha)
			alpha_fs[i] = deconvolved_f
			timer.print_time()

		self.alphas = alphas
		self.alpha_fs = alpha_fs
