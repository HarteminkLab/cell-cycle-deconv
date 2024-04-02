

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.combined_chromatin_model import load_chromatin_model_from_disk


class PTRThresholdSearch:

	def __init__(self, chromatin_dir, gene_name):
		self.chromatin_model = load_chromatin_model_from_disk(gene_name, chromatin_dir)

		# TODO: temporary fix for meta values being incorrect in saving csv
		self.chromatin_model.solver.gamma.value = 0.006

	def compute_ptr(self, q_values):
		self.chromatin_model.compute_ptr(q_values)

	def plot_ptr_and_masks(self, lo=1, hi=6):

		from src.ptr_analysis_plotter import threshold_img

		self.f_ptr_img = self.chromatin_model.f_ptrs.reshape(self.chromatin_model.image_shape)
		(self.cycling_mask, l_mask, h_mask, 
		 h_convolved) = threshold_img(self.f_ptr_img, L=lo, H=hi,
			ret_all=True)

		self.noncycling_mask = self.cycling_mask == 0

		def plot_im_ptr(f_ptr_img):
			plt.imshow(f_ptr_img, origin='lower', cmap='Blues')
			plt.xticks([])
			plt.yticks([])

		# The PTR and threshold plots
		plt.figure(figsize=(18, 5))
		plt.subplot(2, 3, 1)
		plot_im_ptr(self.f_ptr_img)

		plt.subplot(2, 3, 2)
		plot_im_ptr(self.cycling_mask)

		plt.subplot(2, 3, 3)
		plot_im_ptr(self.noncycling_mask)

		plt.subplot(2, 3, 4)
		plot_im_ptr(l_mask)

		plt.subplot(2, 3, 5)
		plot_im_ptr(h_mask)

		plt.subplot(2, 3, 6)
		plot_im_ptr(h_convolved)


	def plot_deconvolved_cycling_mask(self, negative_mask=False):

		if not negative_mask:
			fig = self.chromatin_model.create_deconvolution_plots_abbreviated_flipped(vmax=30,
					mask=self.cycling_mask)
		else:
			fig = self.chromatin_model.create_deconvolution_plots_abbreviated_flipped(vmax=30,
					mask=self.noncycling_mask)
