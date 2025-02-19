

from src.chromatin_entropy import compute_occupancy_entropy, select_sub_img_mid,\
	plot_occupancy_entropy_results
import numpy as np
import pandas as pd
from src.RealDataReplication import read_n_fr_b
from src.chromatin_model import ChromatinModel
from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver


# Chromatin find optimal kappa code

DSE_GENES = ['DSE1', 'DSE2', 'DSE3', 'DSE4']
INTERGENICS = ['4_620908', '6_31764', '5_242410', '4_358326']


def compute_occupancy_entropy_for_file(filepath):
	"""Compute the occupancy and entropy for a given F file"""
	F = np.load(filepath)
	F_imgs = F.reshape((149, 26, -1))
	F_subset = select_sub_img_mid(F_imgs, window=500)
	occupancy_result, entropy_result = compute_occupancy_entropy(F_subset)
	return occupancy_result, entropy_result


def compute_occupancies_entropies_directory(config, directory, names):
	m = config.H.shape[1]

	gene_occupancies = np.zeros((len(names), m))
	gene_entropies = np.zeros((len(names), m))

	for i, name in enumerate(names):
		filepath = f'{directory}/{name}_F.npy'
		occupancy_result, entropy_result = compute_occupancy_entropy_for_file(filepath)
		gene_occupancies[i] = occupancy_result
		gene_entropies[i] = entropy_result

	return gene_occupancies, gene_entropies


def compute_mean_tb_l2(config, gene_occupancies, gene_entropies):

	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')

	def l_norm_t_b(arr, norm=2):
		return np.linalg.norm(arr[t_indices]-arr[b_indices])/len(arr)

	mean_l2_occ = np.apply_along_axis(lambda arr: l_norm_t_b(arr), 1,
											   gene_occupancies).mean()
	mean_l2_entropy = np.apply_along_axis(lambda arr: l_norm_t_b(arr), 1,
											   gene_entropies).mean()

	return mean_l2_occ, mean_l2_entropy


from src.utils import mkdir_safe
from src.chromatin_deconvolution_solver import plot_example_fits
from src.chromatin_deconvolution_solver import plot_branches
from src.chromatin_model import plot_prediction
from src.geneset import get_deconvolved_geneset
from src.sgd import get_orfname
from src.RealDataReplication import read_n_fr_b
from src.chromatin_deconvolution_solver import subset_select_highest_G_indices


def deconvolve_and_plot_gene(config1, gene_name, gamma, kappa, output_directory, f_savepath=None,
					   window=1200, save_plots=True):
	genes = get_deconvolved_geneset()

	orfname = get_orfname(gene_name)

	gene = genes.loc[orfname]
	tss = gene.TSS
	chrom = gene.chr
	padding_2 = window//2
	mnase_span = tss-padding_2, tss+padding_2
	_, N, frep, b = read_n_fr_b(chrom, mnase_span)
	
	chromatin_model = ChromatinModel(config1)
	chromatin_model.load_mnase_span(chrom=chrom, mnase_span=mnase_span)

	# Mean 1
	G_values = chromatin_model.G
	
	chromatin_solver = ChromatinDeconvolveSolver(config1, 
		G_values, N, b, frep)
	full_deconvolved_F = chromatin_solver.deconvolve_G_iteratively(gamma=gamma, 
		kappa=kappa, verbose=False)

	if f_savepath is None:
		f_savepath = f"{output_directory}/{gene_name}_F.npy"

	np.save(f_savepath, full_deconvolved_F)

	if save_plots:
		import matplotlib.pyplot as plt
		fig = plot_branches(chromatin_model, full_deconvolved_F)
		plt.savefig(f"{output_directory}/{gene_name}_branches.png")
		plt.close(fig)

		fig = plot_prediction(chromatin_model, G_values, N, full_deconvolved_F, frep, b)
		plt.savefig(f"{output_directory}/{gene_name}_prediction.png")
		plt.close(fig)

		fig = plot_example_fits(chromatin_solver)
		plt.savefig(f"{output_directory}/{gene_name}_curves.png")
		plt.close(fig)
	
	return chromatin_solver


def deconvolve_and_plot_region(config, chrom, midpoint, gamma, kappa, output_directory,
					   window=1200, f_savepath=None, save_plots=True):

	padding_2 = window//2
	mnase_span = midpoint-padding_2, midpoint+padding_2
	_, N, frep, b = read_n_fr_b(chrom, mnase_span)
	
	chromatin_model = ChromatinModel(config)
	chromatin_model.load_mnase_span(chrom=chrom, mnase_span=mnase_span)

	G_values = chromatin_model.G
	
	chromatin_solver = ChromatinDeconvolveSolver(config, 
		G_values, N, b, frep)
	full_deconvolved_F = chromatin_solver.deconvolve_G_iteratively(gamma=gamma, 
		kappa=kappa, verbose=False)

	region_name = f"{chrom}_{midpoint}"
	
	if f_savepath is None:	
		f_savepath = f"{output_directory}/{region_name}_F.npy"

	np.save(f_savepath, full_deconvolved_F)
	
	if save_plots:
		import matplotlib.pyplot as plt
		fig = plot_branches(chromatin_model, full_deconvolved_F)
		plt.savefig(f"{output_directory}/{region_name}_branches.png")
		plt.close(fig)

		fig = plot_prediction(chromatin_model, G_values, N, full_deconvolved_F, frep, b)
		plt.savefig(f"{output_directory}/{region_name}_prediction.png")
		plt.close(fig)

		fig = plot_example_fits(chromatin_solver)
		plt.savefig(f"{output_directory}/{region_name}_curves.png")
		plt.close(fig)
	
	return chromatin_solver


# Pseudo-code for kappa sweep.


# Loop through genes
# Loop through intergenic regions, index on "name" and "group"

# Create a directory for each
# Define sweep of kappa values (starting with range of values defined in expression)
# Compute deconvolution for each, save the numpy F's to disk
#        *** Save F's to disk, this step may take a long time.
#        *** Save as kappa and gamma values prepended on the filename

# --------

# Compute c-d l2 occupancy and entropy for each deconvolution run
# Compute mean c-d l2 ^
# 
# Compute signal to noise ratio 
# Plot against kappa
#
# Plot the optimal result
#
# aside: Plot arbitraty indices
# aside: Plot all of the occupancy and entropy curves on one plot colored by kappa
