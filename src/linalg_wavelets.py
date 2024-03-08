

import numpy as np



def create_downsampling_matrix(n, factor=2):
	"""
	Create a downsampling matrix to downsample a signal by a given factor.
	"""
	# Number of rows in the downsampling matrix
	rows = int(n // factor + (n % factor > 0))
	
	# Initialize the downsampling matrix with zeros
	down_matrix = np.zeros((rows, n))
	
	# Fill the matrix
	for i in range(rows):
		
		new_dim_i = int(i * factor)

		if new_dim_i >= n: continue

		down_matrix[i, new_dim_i] = 1
	
	return down_matrix


def create_upsampling_matrix(n, factor=2):
	"""Create an upsampling matrix to upsample a signal by a given factor.
	"""
	# Number of columns in the upsampling matrix (length of the upsampled signal)
	cols = n * factor
	# Initialize the upsampling matrix with zeros
	up_matrix = np.zeros((n, cols))
	
	# Fill the matrix
	for i in range(0, n):
		up_matrix[i, i * factor] = 1
	return up_matrix



def create_convolution_matrix(kernel, n, offset=0):
	"""Convert a kernel to a convolution matrix for 'full' mode convolution."""

	k = len(kernel)
	output_length = n

	conv_matrix = np.zeros((output_length, n))

	# Fill in the convolution matrix
	for i in range(0, output_length):
		for j in range(k):

			row = i-k//2+1+offset
			col = (i - j)

			if (0 <= row < output_length) and (0 <= col < n):
				conv_matrix[row, col] = kernel[j]

	return conv_matrix


def create_downsample_convolution_matrix(n, kernel):
	"""
	Create a matrix that will perform a convolution and downsample. This is the 
	deconstruction process for wavelet transformation.

	Returns a (n x n/2) matrix
	"""
	down_matrix = create_downsampling_matrix(n, 2) # (n/2 x n)
	kernel_conv_mat = create_convolution_matrix(kernel, n) # (n x n)
	downsample_conv_mat = np.matmul(kernel_conv_mat, down_matrix.T) # Perform convolution and then downsample
	return downsample_conv_mat


def create_upsample_convolution_matrix(n, kernel):
	"""
	Create a matrix that will upsample and preform a convolution. This matrix will be
	for the reconstruction process of wavelet transformation

	Returns a (n/2 x n) matrix
	"""
	n_2 = n//2
	up_matrix = create_upsampling_matrix(n_2, 2) # (n/2 x n)
	conv_mat = create_convolution_matrix(kernel, n, offset=-1) # (n x n)
	upsampled_conv_mat = np.matmul(up_matrix, conv_mat)
	return upsampled_conv_mat


def is_power_of_two_math(n):
	"""For checking wavelet convolution kernel construction"""
	if n <= 0:
		return False
	log_n = np.log2(n)
	return log_n == int(log_n)

