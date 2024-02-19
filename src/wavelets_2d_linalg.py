

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



# ------

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

# ------------- Kronecker flattened images transformations -----------------


def create_kron_wavelet2d_convolution_matrices(wavelet, input_shape):
	"""Create the wavelet transformation matrices for deconstruction and reconstruction"""

	(decomps_mats, reconst_mats) = create_wavelet2d_convolution_matrices(wavelet, input_shape)

	(down_lo_horizontal_mat, down_lo_vertical_mat, 
	 down_hi_horizontal_mat, down_hi_vertical_mat) = decomps_mats

	(up_lo_horizontal_mat, up_lo_vertical_mat, 
	 up_hi_horizontal_mat, up_hi_vertical_mat) = reconst_mats

	n, m = input_shape

	# Expect the image to be flattened, so along the horizontal dimension, 
	# we use n as our size.
	# along the vertical dimension, the downsample matrix will be half the size
	# so use m//2
	down_lo_kron_horizontal_mat = np.kron(np.eye(n), down_lo_horizontal_mat)
	down_lo_kron_vertical_mat = np.kron(np.eye(m//2), up_lo_vertical_mat)

	down_hi_kron_horizontal_mat = np.kron(np.eye(n), down_hi_horizontal_mat)
	down_hi_kron_vertical_mat = np.kron(np.eye(m//2), down_hi_vertical_mat)


	# We will need tranposition matrices to transpose the flattened image set in
	# place, these images are in vectorized form, but we need a tranpose operation
	# for the vertical transformation steps

	# down tranpose shape
	horizontal_tranpose_shape = n, m//2,

	# up transpose shape
	coeffs_transpose_shape = m//2, n

	up_transpose_mat = create_flattened_transpose_permutation_matrix(coeffs_transpose_shape)
	down_transpose_mat = create_flattened_transpose_permutation_matrix(horizontal_tranpose_shape)

	decomp_kron_mats = (down_lo_kron_horizontal_mat,
						down_lo_kron_vertical_mat,
						down_hi_kron_horizontal_mat,
						down_hi_kron_vertical_mat)

	up_lo_kron_horizontal_mat = np.kron(np.eye(n), up_lo_horizontal_mat)
	up_lo_kron_vertical_mat = np.kron(np.eye(m//2), up_lo_vertical_mat.T)

	up_hi_kron_horizontal_mat = np.kron(np.eye(n), up_hi_horizontal_mat)
	up_hi_kron_vertical_mat = np.kron(np.eye(m//2), up_hi_vertical_mat.T)

	reconst_kron_mats = (up_lo_kron_horizontal_mat,
						up_lo_kron_vertical_mat,
						up_hi_kron_horizontal_mat,
						up_hi_kron_vertical_mat)

	return decomp_kron_mats, reconst_kron_mats
