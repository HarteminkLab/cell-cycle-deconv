
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from cc_src.chromatin_model import ChromatinModel


class CombinedChromatinModel:
	"""
	In this class we will construct the combined chromatin deconvolution model
	"""

	def __init__(self, config1, config2):

		# Rename the config such that when plotting with the title
		# the combined model name is used.
		# Config2 will not be used for plotting
		config1.name = f"Combined, $\\alpha$={config1.alpha},{config2.alpha}"
		config2.name = f"Combined, $\\alpha$={config1.alpha},{config2.alpha}"

		self.chrom1_model = ChromatinModel(config1)
		self.chrom2_model = ChromatinModel(config2)

	def load_combined_mnase_gene(self, gene_name):
		"""This takes the place of load_mnase_gene, as we don't need the
		replicate parameter anymore"""
		self.chrom1_model.load_mnase_gene(gene_name, replicate=1)
		self.chrom2_model.load_mnase_gene(gene_name, replicate=2)

	def	setup_deconv_model(self):
		from src.helpers import calcH
		from src.model import Model

		chrom1_model = self.chrom1_model
		chrom2_model = self.chrom2_model

		# Next we will need to setup the deconvolution model to combine the H
		# and the deconvolution G data
		self.G1 = chrom1_model.deconv_hist
		self.G2 = chrom2_model.deconv_hist

		# Combine the two G datasets row-wise
		self.G = np.concatenate([self.G1, self.G2])

		# Create the first replicates model and H
		self.deconv1_model = Model(chrom1_model.config, chrom1_model.gene_name, chrom1_model.gamma)
		self.H1, self.H1pos = calcH(chrom1_model.config.intervals_wt1, chrom1_model.timepoints)

		# And the second
		self.deconv2_model = Model(chrom2_model.config, chrom2_model.gene_name, chrom2_model.gamma)
		self.H2, self.H2pos = calcH(chrom2_model.config.intervals_wt1, chrom2_model.timepoints)

		# Combine the H matrices
		self.H = np.concatenate([self.H1, self.H2])

		# Use the deconv1 model for deconvolution
		# We shouldn't need anything from model2 at this point
		self.deconv_model = self.deconv1_model
		self.deconv_model.gamma = self.chrom1_model.gamma


	def deconvolve(self):

		from src.timer import Timer
		from src.deconvolve_chromatin import deconvolve_chromatin_H

		print(f"Deconvolving combined model...")

		timer = Timer()
		self.setup_deconv_model()

		# Setup creates two models, in this case we can use either model
		# because the functions we need are related to it and b columns in H and f
		# these should be identical for both replicates
		#
		# Also the gamma value will be built-into this model
		f, rn, sn = deconvolve_chromatin_H(self.deconv_model, self.H, self.G)
		self.set_results(f, rn, sn, self.chrom1_model.gamma)

		print(f"Deconvolved in : {timer.get_time()}")

		print(f"The fitting norm is {self.rn:.2f}, "
			  f"the smoothing norm is: {self.sn:.2f}")


	def deconvolve_find_optimal_gamma(self):
		"""
		Find the optimal gamma value
		"""

		from src.find_gamma_chromatin import FindOptimalGammaChromatin
		from src.timer import Timer

		timer = Timer()

		self.setup_deconv_model()

		# Note that the G data is stored in self.chrom1_model for the find optimal gamma
		# TODO: This needs to be cleaned and made more clear for this combined model
		self.deconv_model.H = self.H

		self.find_gamma_chromatin = FindOptimalGammaChromatin(self.deconv_model, G=self.G)
		self.found_optimal_success = self.find_gamma_chromatin.find_optimal(silence=False)

		self.set_results(self.find_gamma_chromatin.f, 
			self.find_gamma_chromatin.rn, self.find_gamma_chromatin.sn, self.find_gamma_chromatin.gamma)

		print(f"Deconvolved in : {timer.get_time()}")
		print(f"The fitting norm is {self.chrom1_model.rn:.2f}, "
			  f"the smoothing norm is: {self.chrom1_model.sn:.2f}")


	def set_results(self, f, rn, sn, gamma):
		"""Following completion of deconvolution or find gamma deconvolution, we
		will need to set the results to the appropriate fields.

		TODO: This may need to be cleaned up but for now each of the chrom models
		have plotting code individually, so we set the results in each of them
		"""

		self.f = f
		self.rn = rn
		self.sn = sn
		self.gamma = gamma

		self.chrom1_model.f = f
		self.chrom1_model.rn = rn
		self.chrom1_model.sn = sn
		self.chrom1_model.gamma = gamma

		self.chrom2_model.f = f
		self.chrom2_model.rn = rn
		self.chrom2_model.sn = sn
		self.chrom2_model.gamma = gamma

		# Set the f, rn, and sn from the results to the model
		self.pred_G = np.matmul(self.H, f) 
		tp1 = self.chrom1_model.timepoints
		self.pred_G1 = self.pred_G[0:len(tp1)]
		self.pred_G2 = self.pred_G[len(tp1):]


	def create_deconvolution_plots_abbreviated_flipped(self, ge_model=None):
		"""Create the deconvolution plot defined in chromatin_model.py
		"""
		fig = self.chrom1_model.create_deconvolution_plots_abbreviated_flipped(ge_model=ge_model)
		return fig

	def plot_raw_prediction(self, replicate):
		"""Plot the resulting comparison between the raw and predicted data"""

		if replicate == 1:
			title = self.chrom1_model.define_title().replace("Combined", "Combined-Rep.1")
			fig = self.chrom1_model.plot_prediction_comparison(self.pred_G1, title)
		else:
			title = self.chrom1_model.define_title().replace("Combined", "Combined-Rep.2")
			fig = self.chrom2_model.plot_prediction_comparison(self.pred_G2, title)

		return fig
