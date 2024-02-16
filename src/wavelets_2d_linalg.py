

import numpy as np
from src.linalg_wavelets import create_downsample_convolution_matrix, \
	create_upsample_convolution_matrix


def pad_image(image, v_padding, h_padding):
	"""Pad an image by vertical and horizontal padding to handle edge effects for wavelet
	decomposition"""

	padding_horizontal = np.zeros((image.shape[0], h_padding))
	padded_horizontal_image = np.concatenate([padding_horizontal, image, padding_horizontal],
											axis=1)

	padding_vertical = np.zeros((v_padding, padded_horizontal_image.shape[1]))
	padded_image = np.concatenate([padding_vertical, padded_horizontal_image, padding_vertical],
											axis=0)
	return padded_image


def create_wavelet2d_convolution_matrices(wavelet, input_shape):
	"""Create the wavelet transformation matrices for deconstruction and reconstruction"""

	dec_lo, dec_hi = wavelet.dec_lo, wavelet.dec_hi
	rec_lo, rec_hi = wavelet.rec_lo, wavelet.rec_hi

	# Assume a square image for now
	n, m = input_shape

	# ----------- Deconstruction matrices -----------
	down_lo_horizontal_mat = create_downsample_convolution_matrix(m, dec_lo)
	down_lo_vertical_mat = create_downsample_convolution_matrix(n, dec_lo)

	down_hi_horizontal_mat = create_downsample_convolution_matrix(m, dec_hi)
	down_hi_vertical_mat = create_downsample_convolution_matrix(n, dec_hi)

	# --------- Reconstruction matrices -----------
	up_lo_horizontal_mat = create_upsample_convolution_matrix(m, rec_lo)
	up_lo_vertical_mat = create_upsample_convolution_matrix(n, rec_lo)

	up_hi_horizontal_mat = create_upsample_convolution_matrix(m, rec_hi)
	up_hi_vertical_mat = create_upsample_convolution_matrix(n, rec_hi)

	transform_matrices = ((down_lo_horizontal_mat, down_lo_vertical_mat, 
						   down_hi_horizontal_mat, down_hi_vertical_mat),
						  (up_lo_horizontal_mat, up_lo_vertical_mat, 
						   up_hi_horizontal_mat, up_hi_vertical_mat))

	return transform_matrices


def wave2d_decomposition(image, decom_mats):

	(down_lo_horizontal_mat, down_lo_vertical_mat, 
	 down_hi_horizontal_mat, down_hi_vertical_mat) = decom_mats

	L = image@down_lo_horizontal_mat
	H = image@down_hi_horizontal_mat

	LL = down_lo_vertical_mat.T@L
	HL = down_lo_vertical_mat.T@H
	LH = down_hi_vertical_mat.T@L
	HH = down_hi_vertical_mat.T@H

	return LL, HL, LH, HH


def wave2d_reconstruction(coeffs, recon_mats):

	(up_lo_horizontal_mat, up_lo_vertical_mat, 
	 up_hi_horizontal_mat, up_hi_vertical_mat) = recon_mats

	(LL, HL, LH, HH) = coeffs

	LL_reconstruction = up_lo_vertical_mat.T@LL@up_lo_horizontal_mat
	HH_reconstruction = up_hi_vertical_mat.T@HH@up_hi_horizontal_mat
	LH_reconstruction = up_hi_vertical_mat.T@LH@up_lo_horizontal_mat
	HL_reconstruction = up_lo_vertical_mat.T@HL@up_hi_horizontal_mat

	reconstruction = LL_reconstruction+HH_reconstruction+LH_reconstruction+HL_reconstruction

	return reconstruction
