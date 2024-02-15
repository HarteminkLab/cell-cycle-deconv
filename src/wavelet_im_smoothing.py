
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt


class ChromatinWaveletsComparison():
	"""A class to compare different families of wavelets and thresholding values. Requires a chromatin model
	to test the idea to easily retrieve a chromatin image."""

	def __init__(self, chromatin_model):
		self.chromatin_model = chromatin_model

	def create_coefficients(self, wavelet_name='db1', thresholds_low=np.linspace(0, 4, 5), 
		thresholds_hi=np.linspace(0, 4, 5)):

		self.wavelet_name = wavelet_name
		self.thresholds_low = thresholds_low
		self.thresholds_hi = thresholds_hi

		wavelet_generators = []

		for i in range(len(thresholds_low)):

			cur_gens = []

			for j in range(len(thresholds_hi)):

				img = self.chromatin_model.deconv_hist_unflattened[6]
				wavelets = SpatialWavelets(img, wavelet_name)
				wavelets.compute_coeffs()
				wavelets.apply_threshold(thresholds_low[i], thresholds_hi[j])
				wavelets.reconstruct_image()	
				cur_gens.append(wavelets)

			wavelet_generators.append(cur_gens)

		
		self.wavelet_generators = wavelet_generators

	def plot_thresholds(self):

		n = len(self.wavelet_generators)
		m = len(self.wavelet_generators[0])

		fig, axs = plt.subplots(n, m, figsize=(9, 4))
		plt.subplots_adjust(top=0.8)

		for i in range(n):

			for j in range(m):

				ax = axs[i][j]

				if i == 0:
					ax.set_title(f"Appr: {self.thresholds_low[j]}", rotation=0, ha='right')

				gen = self.wavelet_generators[i][j]
				gen.plot_img(gen.reconstructed_image, ax)

				if j == 0:
					ax.set_ylabel(f"Detail: {self.thresholds_hi[i]}", rotation=0, ha='right')

		plt.suptitle("Filter: " + self.wavelet_name)


class SpatialWavelets():

	def __init__(self, img, wavelet_name='db1'):
		self.img = img.copy()
		self.wavelet_name = wavelet_name
		

	def plot_img(self, img0, ax=None):
		if ax is None: 
			fig = plt.figure(figsize=(3, 1))
			ax = plt.gca()
		im = ax.imshow(img0, aspect='auto', origin='lower', cmap='magma_r',
			vmax=10)
		# plt.colorbar(im)
		ax.set_xticks([])
		ax.set_yticks([])

	def compute_coeffs(self):

		# Perform 2D Wavelet Transformation
		self.coeffs = pywt.dwt2(self.img, self.wavelet_name)


	def apply_threshold(self, threshold_LL, threshold_detail):

		coeffs = self.coeffs

		# Process each level of coefficients
		thresholded_coeffs = []
		for coeff in coeffs:
			if isinstance(coeff, tuple):
				# For detail coefficients, apply threshold
				thresholded_coeffs.append(tuple(np.where(np.abs(c) > threshold_detail, c, 0) for c in coeff))
			else:

				thresholded_coeffs.append(np.where(np.abs(coeff) > threshold_LL, coeff, 0))

				# For approximation coefficients, you might choose not to threshold
				# Or apply a different strategy
				# thresholded_coeffs.append(coeff)

		self.coeffs = thresholded_coeffs

	def how_many_non_zero_coeffs(self):
		LL, (HL, LH, HH) = self.coeffs

		print("Non-zero LL, HL, LH, HH coefficients:\t", (LL > 0).sum(), end="\t")
		print((HL > 0).sum(), end="\t")
		print((LH > 0).sum(), end="\t")
		print((HH > 0).sum())

	def plot_coefficients(self):
		def _plot_coeff(c):
			plt.imshow(c, aspect='auto', origin='lower', cmap='magma_r')
			plt.xticks([])
			plt.yticks([])

		LL, (HL, LH, HH) = self.coeffs
		plt.figure(figsize=(12, 1))
		plt.subplot(1, 4, 1)
		_plot_coeff(LL)

		plt.subplot(1, 4, 2)
		_plot_coeff(HL)

		plt.subplot(1, 4, 3)
		_plot_coeff(LH)

		plt.subplot(1, 4, 4)
		_plot_coeff(HH)


	def coefficients_matrix(self):
		"""Return the image's coefficients as a matrix, for easier manipulation


		Returns a (4, rows, columns) matrix representing the LL, HL, LH, HH components of
		the wavelet transformation
		"""
		
		LL, (HL, LH, HH) = self.coeffs

		coefficients = np.zeros((4, *LL.shape))
		coefficients[0] = LL
		coefficients[1] = HL
		coefficients[2] = LH
		coefficients[3] = HH

		return coefficients


	def plot_comparison(self, ax0=None, ax1=None):

		if ax0 is None:
			fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 1))

		self.plot_img(self.img, ax0) 
		self.plot_img(self.reconstructed_image, ax1)

	def reconstruct_image(self):
		# Perform Inverse 2D Wavelet Transformation
		self.reconstructed_image = pywt.idwt2(self.coeffs, self.wavelet_name)
		self.reconstructed_image[self.reconstructed_image < 0] = 0
