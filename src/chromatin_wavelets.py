
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.wavelet_im_smoothing import SpatialWavelets


class ChromatinWavelets():
	"""A class to handle the conversion of chromatin images from a chromatin model into coefficients space, vectorize
	the coefficients space for deconvolution, and reversing the transformations to represent the reconstructed images
	"""

	def __init__(self, chromatin_model):
		self.chromatin_model = chromatin_model


	def create_coefficients(self, wavelet_name='bior4.4'):

		self.wavelet_name = wavelet_name

		n = len(self.chromatin_model.timepoints)

		self.wavelet_generators = []
		self.coefficients_matrix = None

		for i in range(n):

			img = self.chromatin_model.deconv_hist_unflattened[i]
			wavelets = SpatialWavelets(img, wavelet_name)
			wavelets.compute_coeffs()
			wavelets.apply_threshold(2, 2)
			coeffs_mat = wavelets.coefficients_matrix()

			# Lazy load to get the matrix dimensions
			if self.coefficients_matrix is None:
				self.coefficients_matrix = np.zeros((n, *coeffs_mat.shape))

			self.coefficients_matrix[i] = coeffs_mat
			self.wavelet_generators.append(wavelets)

		self.create_coefficients_vectors()

		print(f"Created coefficients matrix of shape: {self.coefficients_matrix.shape}")
		print(f"Created coefficients vectorized matrix of shape: {self.coeffs_vec.shape}")


	def create_coefficients_vectors(self):
		"""This method will convert the coefficients matrix into a matrix of vectors.

		In that, the coefficients matrix is originally in the form of:


			(number of timepoints, number of coefficients, num rows of coeffs, num cols of coeffs)


		We are interested in:

			(number of timepoints, number of all coefficients)

		Because we will be deconvolving a  2D matrix where the rows are the timepoints and the columns are the set of
		values we want to deconvolve.

		After deconvolution, we will convert the matrix back into the coefficients form for reconstruction.

		"""

		coeffs_mat = self.coefficients_matrix

		# Vectorized coefficients matrix
		n = coeffs_mat.shape[0]
		m = coeffs_mat[0].reshape(-1).shape[0]

		coeffs_vec = np.zeros((n, m))

		for i in range(n):
			cur_coeffs = coeffs_mat[i]
			cur_coeffs_vec = cur_coeffs.reshape(-1)
			coeffs_vec[i] = cur_coeffs_vec

		self.coeffs_vec = coeffs_vec

	def convert_vectorized_coeffs_to_mat(self, vec_mat):
		"""Convert the vectorized coefficients to matrix form for reconstruction.

		This will be used for the deconvolved f matrix.
		"""

		u = vec_mat.shape[0]
		original_mat_shape = self.coefficients_matrix.shape
		ret_mat = vec_mat.reshape((u, original_mat_shape[1], original_mat_shape[2], original_mat_shape[3]))

		return ret_mat


	def reconstruct_images(self, coeffs_mat):

		n = coeffs_mat.shape[0]
		reconstructed_images = None

		for i in range(n):
			cur_coeffs = coeffs_mat[i]
			coeffs = cur_coeffs[0], (cur_coeffs[1], cur_coeffs[2], cur_coeffs[3])
			reconstructed_image = pywt.idwt2(coeffs, self.wavelet_name)

			if reconstructed_images is None:
				reconstructed_images = np.zeros((n, *reconstructed_image.shape))

			reconstructed_images[i] = reconstructed_image

		return reconstructed_images

