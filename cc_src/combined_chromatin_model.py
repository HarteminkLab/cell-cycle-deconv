
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from cc_src.chromatin_model import ChromatinModel


class CombinedChromatinModel(ChromatinModel):
	"""
	In this class we will construct the combined chromatin deconvolution model
	"""

	def __init__(self, config1, config2):
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
		deconv_model = self.deconv1_model
		deconv_model.gamma = self.gamma

		self.f, self.rn, self.sn = deconvolve_chromatin_H(deconv_model, self.H, self.G)

		print(f"Deconvolved in : {timer.get_time()}")

		print(f"The fitting norm is {self.rn:.2f}, "
			  f"the smoothing norm is: {self.sn:.2f}")

	def create_deconvolution_plots_abbreviated_flipped(self):

		self.chrom1_model.f = self.f
		self.chrom1_model.rn = self.rn
		self.chrom1_model.sn = self.sn

		fig = self.chrom1_model.create_deconvolution_plots_abbreviated_flipped()
