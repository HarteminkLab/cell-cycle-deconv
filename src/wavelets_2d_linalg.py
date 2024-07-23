

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
	up_hi_horizontal_mat = create_upsample_convolution_matrix(m, rec_hi)

	up_lo_vertical_mat = create_upsample_convolution_matrix(n, rec_lo)
	up_hi_vertical_mat = create_upsample_convolution_matrix(n, rec_hi)

	transform_matrices = ((down_lo_horizontal_mat, down_lo_vertical_mat, 
						   down_hi_horizontal_mat, down_hi_vertical_mat),
						  (up_lo_horizontal_mat, up_lo_vertical_mat, 
						   up_hi_horizontal_mat, up_hi_vertical_mat))

	return transform_matrices



def wave2d_decomposition_kron(images, decom_kron_mats):
	"""Decomposition against flattened images"""

	(down_lo_horizontal_mat, down_lo_vertical_mat, 
	 down_hi_horizontal_mat, down_hi_vertical_mat) = decom_kron_mats

	L = images@down_lo_horizontal_mat
	H = images@down_hi_horizontal_mat

	LL = down_lo_vertical_mat.T@L
	HL = down_lo_vertical_mat.T@H
	LH = down_hi_vertical_mat.T@L
	HH = down_hi_vertical_mat.T@H

	return LL, HL, LH, HH


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

	# Equivalent method to keep input on the left
	#LL_reconstruction = (LL.T@up_lo_vertical_mat).T@up_lo_horizontal_mat

	LL_reconstruction = up_lo_vertical_mat.T@LL@up_lo_horizontal_mat
	HH_reconstruction = up_hi_vertical_mat.T@HH@up_hi_horizontal_mat
	LH_reconstruction = up_hi_vertical_mat.T@LH@up_lo_horizontal_mat
	HL_reconstruction = up_lo_vertical_mat.T@HL@up_hi_horizontal_mat

	reconstruction = LL_reconstruction+HH_reconstruction+LH_reconstruction+HL_reconstruction

	return reconstruction


# ------------- Kronecker flattened images transformations -----------------

# The following functions define the matrix transformations to allow us to decompose and 
# reconstruct a flattened
# set of images. 
#
# Motivation is that in cvxpy, we can only operate on 2D matrices. Thus, we use the first dimension 
# for our stack of images, and the second as our flattened width and height image.
#
# The trick here is using the Kronecker matrix transformation and transpose permutation matrix.
#
# These matrices allow us to operate on the flattened images and perform matrix 
# multiplication operations as if the image was unflattened. In this way we can perform the 
# decomposition and reconstruction operations on the input
# image without the need to loop or reshape the input image stack.
#

def create_flattened_transpose_permutation_matrix(shape):
	"""Creates a permutation matrix such that we can transpose the 
	elements in a flattened matrix without needing to reshape the vector
	back into its original matrix form.
	
	This will be useful when we need to perform matrix multiplication transformations
	on our flattened image set and we need to transpose our images. We will not be able to
	do that directly because our first dimension is now the size of our image stack.
	
	Rather, we are going to transpose the vectorized form of the images in place with
	a permutation matrix.
	"""
	n, m = shape
	total_elements = n * m
	P = np.zeros((total_elements, total_elements))
	
	for i in range(n):
		for j in range(m):
			src_index = i * m + j
			target_index = j * n + i
			P[src_index, target_index] = 1
			
	return P

def create_kron_wavelet2d_convolution_matrices(wavelet, input_shape, preprocessing_mat=None):
	"""Create the wavelet transformation matrices for decomposition and reconstruction.

	After loading the standard wavelet transformation matrices, convert them 
	into kronecker transformation
	matrices.

	Then, we have operations in which we need the input on the far left side, 
	so use the transpose permutation matrices to enforce this rule.
	"""

	(decomps_mats, reconst_mats) = create_wavelet2d_convolution_matrices(wavelet, input_shape)

	(down_lo_horizontal_mat, down_lo_vertical_mat, 
	 down_hi_horizontal_mat, down_hi_vertical_mat) = decomps_mats

	(up_lo_horizontal_mat, up_lo_vertical_mat, 
	 up_hi_horizontal_mat, up_hi_vertical_mat) = reconst_mats

	n, m = input_shape

	# Each step downsamples by half, so the vertical matrices will take in
	# half of the row width
	down_lo_kron_horizontal_mat = np.kron(np.eye(n), down_lo_horizontal_mat)
	down_hi_kron_horizontal_mat = np.kron(np.eye(n), down_hi_horizontal_mat)

	down_lo_kron_vertical_mat = np.kron(np.eye(m//2), down_lo_vertical_mat)
	down_hi_kron_vertical_mat = np.kron(np.eye(m//2), down_hi_vertical_mat)

	decomp_kron_mats = (down_lo_kron_horizontal_mat,
						down_lo_kron_vertical_mat,
						down_hi_kron_horizontal_mat,
						down_hi_kron_vertical_mat)

	# Reverse the process, upsample vertically, than horizontally
	up_lo_kron_vertical_mat = np.kron(np.eye(m//2), up_lo_vertical_mat)
	up_hi_kron_vertical_mat = np.kron(np.eye(m//2), up_hi_vertical_mat)
	up_lo_kron_horizontal_mat = np.kron(np.eye(n), up_lo_horizontal_mat)
	up_hi_kron_horizontal_mat = np.kron(np.eye(n), up_hi_horizontal_mat)

	reconst_kron_mats = (up_lo_kron_horizontal_mat,
						up_lo_kron_vertical_mat,
						up_hi_kron_horizontal_mat,
						up_hi_kron_vertical_mat)

	hori_T_shape = n, m//2,
	vert_T_shape = m//2, n
	up_transpose_mat = create_flattened_transpose_permutation_matrix(vert_T_shape)
	down_transpose_mat = create_flattened_transpose_permutation_matrix(hori_T_shape)

	# The vertical downsampling transpose operation
	# The horizontal downsampling transpose operation
	t_down_lo_vertical_mat = down_transpose_mat @ down_lo_kron_vertical_mat
	t_down_hi_vertical_mat = down_transpose_mat @ down_hi_kron_vertical_mat

	t_up_lo_horizontal_mat = up_transpose_mat @ up_lo_kron_horizontal_mat
	t_up_hi_horizontal_mat = up_transpose_mat @ up_hi_kron_horizontal_mat

	# The output kronecker matrices
	decomp_LL_mat = down_lo_kron_horizontal_mat @ t_down_lo_vertical_mat
	decomp_LH_mat = down_lo_kron_horizontal_mat @ t_down_hi_vertical_mat
	decomp_HL_mat = down_hi_kron_horizontal_mat @ t_down_lo_vertical_mat
	decomp_HH_mat = down_hi_kron_horizontal_mat @ t_down_hi_vertical_mat

	recomp_LL_mat = up_lo_kron_vertical_mat @ t_up_lo_horizontal_mat
	recomp_LH_mat = up_hi_kron_vertical_mat @ t_up_lo_horizontal_mat
	recomp_HL_mat = up_lo_kron_vertical_mat @ t_up_hi_horizontal_mat
	recomp_HH_mat = up_hi_kron_vertical_mat @ t_up_hi_horizontal_mat

	kron_decomposition_mats = (
		decomp_LL_mat,
		decomp_LH_mat,
		decomp_HL_mat,
		decomp_HH_mat
	)

	kron_reconstruction_mats = (
		recomp_LL_mat,
		recomp_HL_mat,
		recomp_LH_mat,
		recomp_HH_mat,
	)

	return kron_decomposition_mats, kron_reconstruction_mats


def decompose_flattened_kron_coeffs(flattened_images, kron_decomposition_mats):
	"""Take in as input, a matrix of flattened images (n x (height*width)) and the
	decomposition matrices defined in `create_kron_wavelet2d_convolution_matrices`

	Output a tuple of coefficients for each of the wavelet transformation coefficients.
	"""

	(decomp_LL_mat, decomp_LH_mat,
	 decomp_HL_mat, decomp_HH_mat) = kron_decomposition_mats

	LL = flattened_images @ decomp_LL_mat
	LH = flattened_images @ decomp_LH_mat
	HL = flattened_images @ decomp_HL_mat
	HH = flattened_images @ decomp_HH_mat

	return LL, LH, HL, HH


def reconstruct_flattened_kron_coeffs(coeffs, kron_reconstruction_mats):
	"""Reconstruct the original flattened images from the coefficients.

	Input the coefficients tuple and reconstruction matrices define
	in `create_kron_wavelet2d_convolution_matrices`.

	And output the reconstructed flattened images
	"""

	(LL, LH, HL, HH) = coeffs

	(LL_recon_mat, LH_recon_mat,
	 HL_recon_mat, HH_recon_mat) = kron_reconstruction_mats

	reconstructed_LL = LL @ LL_recon_mat
	reconstructed_LH = LH @ LH_recon_mat
	reconstructed_HL = HL @ HL_recon_mat
	reconstructed_HH = HH @ HH_recon_mat

	# Combine to create the reconstructed image
	reconstructed = reconstructed_LL + reconstructed_LH + reconstructed_HL + reconstructed_HH

	return reconstructed
