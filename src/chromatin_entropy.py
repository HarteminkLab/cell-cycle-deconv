
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d


def select_sub_img_mid(f_imgs, window=500):
    """Helper function to subselect a window of given size along the center
    of the parent window.
    
    Input is an np array of f images (t, r, c).
    Outputs (t, r, c'), where c' is length of the new window
    """
    midbin_idx = f_imgs.shape[2]//2
    window = 500
    win_2 = window//2
    win_2_bins = win_2//10
    bin_span = midbin_idx-win_2_bins, midbin_idx+win_2_bins
    f_subset = f_imgs[:, :, bin_span[0]:bin_span[1]]
    return f_subset


# Functions to compute chromatin entropy
def compute_occupancy_entropy(f_images, kernel=None):
    """Compute the occupancy and entropy for a window, given a kernel
    that can subselect read lengths, default is a kernel that selects all
    fragment lengths along a column."""
        
    t, r, c = f_images.shape

    if kernel is None:
        kernel = np.ones((r, 1))
        
    kr, kc = kernel.shape
    out_r = r - kr + 1
    out_c = c - kc + 1
        
    # Pre-allocate output array
    occupancy_result = np.zeros((t, out_r, out_c))
    entropy_scores = np.zeros(t)
    eps = 1e-5
    
    from src.helpers import calc_entropy

    for i in range(t):
        occupancy_result[i] = convolve2d(f_images[i], kernel, mode='valid')        

        entropy_scores[i] = \
            np.apply_along_axis(lambda row: calc_entropy(row+eps), axis=1, 
            arr=occupancy_result[i])

    return occupancy_result, entropy_scores


def plot_occupancy_entropy_results(config, occupancy_result, entropy_result):

	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')

	plt.figure(figsize=(9, 3))
	plt.subplot(1, 2, 1)
	plt.plot(occupancy_result[t_indices].mean(axis=2), label="Top")
	plt.plot(occupancy_result[b_indices].mean(axis=2), label="Bottom")
	plt.title("Nucleosome occupancy")
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(entropy_result[t_indices], label="Top")
	plt.plot(entropy_result[b_indices], label="Bottom")
	plt.title("Nucleosome disorganization (entropy)")
	plt.legend()
	plt.ylim(4.2, 5.6)

	# Room for suptitle
	plt.subplots_adjust(top=0.8)
