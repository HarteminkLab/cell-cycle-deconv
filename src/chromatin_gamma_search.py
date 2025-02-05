import numpy as np
from matplotlib import pyplot as plt
from src.helpers import get_wavelet_kernel
from matplotlib.colors import ListedColormap

def create_gamma_sweep_plots_single_measure(config, N, H, Fs, f_rep, gamma_sweep, G,
	ylims=(0, 1)):

	i_indices = config.get_Hpositions_for_branch('i')
	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')

	i_timepoints = config.get_timepoints_for_branch('i')
	t_timepoints = config.get_timepoints_for_branch('t')

	plt.figure(figsize=(11, 6))
	plt.subplot(2, 2, 1)

	tb_indices = np.concatenate([t_indices, b_indices])

	W_i = get_wavelet_kernel(len(i_indices))
	W_tb = get_wavelet_kernel(len(tb_indices))

	cmap = ListedColormap(plt.cm.inferno(np.linspace(0.25, 0.85, 256)))

	for i in range(Fs.shape[0]):
		F = Fs[i]
		smoothness_i = W_i@F[i_indices]
		plt.plot(smoothness_i, c=cmap(i/len(Fs)))

	for i in range(Fs.shape[0]):
		F = Fs[i]
		smoothness_tb = W_tb@F[tb_indices]
		plt.plot(smoothness_tb, c=cmap(i/len(Fs)))

	plt.title("Wavelet coefficients")
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gamma_sweep.min(),
		vmax=gamma_sweep.max()))
	W_len = W_i.shape[0]
	plt.xticks([0.15*W_len, 0.85*W_len], ["Lower frequency", "Higher frequency"])
	plt.xlim(0, W_len)
	plt.gca().tick_params(axis='x', length=0)
	plt.ylim(-0.4, 0.4)

	plt.subplot(2, 2, 2)
	for i in range(Fs.shape[0]):
		prediction = N@config.H@np.multiply(Fs[i], f_rep)
		plt.plot(config.timepoints, prediction, c=cmap(i/len(Fs)))
	plt.plot(config.timepoints, G, c='black', lw=4, label="Raw data")
	plt.legend()
	plt.title("Goodness of fit")
	plt.ylim(*ylims)
	plt.xlabel("Experiment time, min")
	plt.xlim(0, config.timepoints[-1])

	cbar = plt.colorbar(sm)
	cbar.ax.set_ylabel('Smoothness, $\\gamma$', rotation=270, va='bottom')

	plt.subplot(2, 2, 3)
	for i in range(Fs.shape[0]):
		plt.plot(i_timepoints, Fs[i, i_indices], c=cmap(i/len(Fs)))
	plt.title("Initial branch")
	plt.ylim(*ylims)
	plt.xlim(0, 60)
	plt.xlabel("Average single cell cycle time, min")

	plt.subplot(2, 2, 4)
	for i in range(Fs.shape[0]):
		plt.plot(t_timepoints, Fs[i, t_indices], c=cmap(i/len(Fs)))
	plt.title("Top branch")
	plt.ylim(*ylims)
	plt.xlim(0, 60)
	plt.xlabel("Average single cell cycle time, min")

	plt.subplots_adjust(hspace=0.6, top=0.86)

	plt.suptitle("Gamma sweep of single chromatin metric", fontsize=16)


import numpy as np
from scipy.signal import savgol_filter

def find_elbow_point(rn, sn, gamma, window_length=6, polyorder=2):
    """Compute the optimal elbow in the curve point with smoothed derivatives."""
    
    # Normalize values
    rn_norm = (rn - rn.min()) / (rn.max() - rn.min()) 
    sn_norm = (sn - sn.min()) / (sn.max() - sn.min())
    
    # Apply Savitzky-Golay filter to smooth the normalized data
    rn_smooth = savgol_filter(rn_norm, window_length, polyorder)
    sn_smooth = savgol_filter(sn_norm, window_length, polyorder)
    
    # Compute first derivatives with smoothing
    drn = savgol_filter(rn_smooth, window_length, polyorder, deriv=1)
    dsn = savgol_filter(sn_smooth, window_length, polyorder, deriv=1)
    
    # Compute second derivatives with smoothing
    d2rn = savgol_filter(rn_smooth, window_length, polyorder, deriv=2)
    d2sn = savgol_filter(sn_smooth, window_length, polyorder, deriv=2)
    
    # Compute curvature 
    numerator = np.abs(drn * d2sn - dsn * d2rn)
    denominator = (drn**2 + dsn**2)**(3/2)
    
    # Avoid division by very small numbers
    mask = denominator > 1e-10
    curvature = np.zeros_like(rn)
    curvature[mask] = numerator[mask] / denominator[mask]
    
    optimal_index = np.argmax(curvature)
    
    return (rn[optimal_index], sn[optimal_index], 
            gamma[optimal_index], optimal_index, 
            curvature)

