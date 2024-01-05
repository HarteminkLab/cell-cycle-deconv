
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp


def load_yl_chromatin_data(orf_name):

	# Load the chromatin data
	prom_sm_occ = pd.read_csv('datasets/yl_rep2_occupancy_for_deconv_2023_10_06/'\
							  'chrom_yl_rep2_prom_sm_occ.tsv', header=None,
							  sep='\t')
	orf_names = pd.read_csv('datasets/yl_rep2_occupancy_for_deconv_2023_10_06/orf_names.csv', 
							header=None)[0].values
	prom_sm_occ.index = orf_names

	nuc_gb_occ = pd.read_csv('datasets/yl_rep2_occupancy_for_deconv_2023_10_06/'\
							 'chrom_yl_rep2_gb_nuc_occ.tsv', header=None,
							  sep='\t')
	nuc_gb_occ.index = orf_names

	# Load for the gene
	prom_g = prom_sm_occ.loc[orf_name].values
	nuc_g = nuc_gb_occ.loc[orf_name].values

	# Stack the promoter small fragments and gene body nucleosomes
	# into a matrix of dimension: (time x 2)
	chromatin_g = np.stack([prom_g, nuc_g], axis=1)

	# Drop the 110 timepoint (4th from the end). 
	# TODO: I'm not sure if we still need to do this
	# But the gene expression data has it dropped. We can come back to this later...
	chromatin_g = np.concatenate([chromatin_g[:-4], chromatin_g[-3:]])

	# print("The shape of the chromatin matrix, G is: ", chromatin_g.shape)

	return chromatin_g


def plot_deconvolved_chromatin(H, f, g):
	# Plot the result
	predicted_G = np.matmul(H, f)

	# Thus, this simple example is resolved.
	plt.figure(figsize=(6, 2))
	plt.subplot(1, 2, 1)
	plt.plot(predicted_G[:, 0])
	plt.title("The raw data 1")
	plt.plot(g[:, 0])
	plt.title("The fit 1, H*f")

	plt.subplot(1, 2, 2)
	plt.plot(predicted_G[:, 1])
	plt.title("The raw data 2")

	plt.plot(g[:, 1])
	plt.title("The fit 2, H*f")

	

def deconvolve_chromatin(model, g):
	"""
	Deconvolve the chromatin array


	TODO: this is nearly identical
	to the Model.deconvolve() method.
	So we may want to refactor to just have one method 
	"""


	# We will add a very small value to g, to avoid divide by zero errors
	eps = 1e-5
	g = g + eps


	H = model.H
	gamma = model.gamma
	factor_fb = 1.5

	# And we are trying to determine the best f (4x1) that
	# minimizes Hxf / g - 1, (4x4) * (4x2) / (4x2) - 1
	f = cp.Variable((H.shape[1], g.shape[1]))

	from src.helpers import get_wavelet_kernel

	f_it = model.get_f_it()	
	f_b = model.get_f_b()

	# Mirroring
	f_b_mirror = np.concatenate((f_b, f_b))
	f_it_mirror = np.concatenate((f_it, np.flip(f_it)))
	factor_fb = 1.5

	W1 = get_wavelet_kernel(len(f_it_mirror))
	W2 = get_wavelet_kernel(len(f_b))
	W2pad = np.zeros((len(f_b), len(f_b)))
	W2 = np.concatenate((np.concatenate((W2, W2pad)), 
						 np.concatenate((W2pad, np.fliplr(W2)))), axis=1)

	# Add the wavelet smoothing constraint to the convex optimization
	objective = cp.Minimize(cp.square(cp.pos(cp.norm(H@f/g - 1))) 
		+ gamma * (cp.norm(W1@f[f_it_mirror], 1) 
		+ factor_fb * cp.norm(W2@f[f_b_mirror], 1))/g.mean())

	# Where f is non-negative
	constraints = [f >= 0]

	# Perform the convex optimization
	prob = cp.Problem(objective, constraints)
	result = prob.solve(solver=cp.CLARABEL)

	# ------- Compute the smoothing norm and fitting/residual norms --------------

	# We will use the non-mirrored wavelet kernel sizes, because we are operating on the 
	# final f values
	W1 = get_wavelet_kernel(len(f_it))
	W2 = get_wavelet_kernel(len(f_b))

	model.sn = (np.linalg.norm(np.matmul(W1, f.value[f_it]), 1) +
	 np.linalg.norm(np.matmul(W2, f.value[f_b]), 1)) / np.mean(g)

	model.rn = np.square(np.clip(np.linalg.norm(np.matmul(model.H, f.value) / g - 1), 0, None))
	model.f = f.value

	return result, f.value
