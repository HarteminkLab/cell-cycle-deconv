

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

	L = np.matmul(image, down_lo_horizontal_mat)
	H = np.matmul(image, down_hi_horizontal_mat)

	LL = np.matmul(L.T, down_lo_vertical_mat).T
	HL = np.matmul(H.T, down_lo_vertical_mat).T
	LH = np.matmul(L.T, down_hi_vertical_mat).T
	HH = np.matmul(H.T, down_hi_vertical_mat).T

	return LL, HL, LH, HH

def wave2d_reconstruction(coeffs, recon_mats):

	(up_lo_horizontal_mat, up_lo_vertical_mat, 
	 up_hi_horizontal_mat, up_hi_vertical_mat) = recon_mats

	(LL, HL, LH, HH) = coeffs

	upsampled_vertical_LL = np.matmul(LL.T, up_lo_vertical_mat).T
	upsampled_vertical_HH = np.matmul(HH.T, up_hi_vertical_mat).T
	upsampled_vertical_HL = np.matmul(HL.T, up_lo_vertical_mat).T
	upsampled_vertical_LH = np.matmul(LH.T, up_hi_vertical_mat).T

	reconstructed_LL = np.matmul(upsampled_vertical_LL, up_lo_horizontal_mat)
	reconstructed_HH = np.matmul(upsampled_vertical_HH, up_hi_horizontal_mat)
	reconstructed_LH = np.matmul(upsampled_vertical_LH, up_lo_horizontal_mat)
	reconstructed_HL = np.matmul(upsampled_vertical_HL, up_hi_horizontal_mat)

	reconstructed = reconstructed_LL + reconstructed_LH + reconstructed_HL + reconstructed_HH + reconstructed_HL

	return reconstructed
