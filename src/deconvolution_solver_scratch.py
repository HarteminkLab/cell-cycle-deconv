


# Scratch space for different types of smoothign





		# smooth_f_i_result = W_i@f_padded[f_i_padded]
		# smooth_f_t_result = W_t@f_padded[f_t_padded]






			# + self.gamma * cp.sum(cp.abs(smooth_f_i_result))/self.g.mean() * 2
			# + self.gamma * cp.sum(cp.abs(smooth_f_t_result))/self.g.mean()  





		
		# f_t = np.concatenate([f_t, f_t])

		# # How much padding?
		# # Double the size of the wavelet and append and prepend
		# padding_i = 0#len(f_i) 
		# padding_t = 0#len(f_t)
		# padding = padding_i + padding_t

		# f_fit_indices = np.arange(m)
		# f_padding_indices = np.arange(m, m+padding)
		# f_padded = cp.Variable(m+padding)

		# f_i_padding_indices = f_padding_indices[:padding_i]
		# f_t_padding_indices = f_padding_indices[padding_i:(padding_i+padding_t)]

		# f_i_padded = np.concatenate([f_i_padding_indices[:padding_i//2], 
		#  	f_i, f_i_padding_indices[padding_i//2:]])

		# f_t_padded = np.concatenate([f_t_padding_indices[:padding_t//2], 
		#  	f_t, f_t_padding_indices[padding_t//2:]])

		# W_i = get_wavelet_kernel(len(f_i_padded))
		# W_t = get_wavelet_kernel(len(f_t_padded))