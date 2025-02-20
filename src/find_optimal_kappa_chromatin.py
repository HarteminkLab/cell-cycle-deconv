

from src.chromatin_entropy import compute_occupancy_entropy, select_sub_img_mid,\
	plot_occupancy_entropy_results
import numpy as np
import pandas as pd
from src.RealDataReplication import read_n_fr_b
from src.chromatin_model import ChromatinModel
from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver
from src.sgd import get_orfname
import matplotlib.pyplot as plt


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


def compute_occupancies_entropies_directory(config, directory, names, file_format="{}_F.npy"):
	m = config.H.shape[1]

	gene_occupancies = np.zeros((len(names), m))
	gene_entropies = np.zeros((len(names), m))

	for i, name in enumerate(names):
		filepath = file_format.format(name)
		full_path =  f'{directory}/{filepath}'
		occupancy_result, entropy_result = compute_occupancy_entropy_for_file(full_path)
		gene_occupancies[i] = occupancy_result
		gene_entropies[i] = entropy_result

	return gene_occupancies, gene_entropies


def l_norm_t_b(config, arr, norm=2):
	t_indices = config.get_Hpositions_for_branch('t')
	b_indices = config.get_Hpositions_for_branch('b')
	return np.linalg.norm(arr[t_indices]-arr[b_indices])/len(arr)


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


def compute_occupancies_entropies_l2s(config1, meta_df):

	# Dataframe of occupancy values
	occupancy_df = pd.DataFrame(columns=np.arange(config1.H.shape[1]),
							   index=meta_df.index)
	entropies_df = pd.DataFrame(columns=np.arange(config1.H.shape[1]),
							   index=meta_df.index)
	l2s_df = meta_df.copy()

	for name, row in meta_df.iterrows():

		full_path = row.path
		gamma = row.gamma

		# Compute the entropy and occupancy scores for each of the runs
		occupancy_result, entropy_result = compute_occupancy_entropy_for_file(full_path)

		occupancy_df.loc[name, :] = occupancy_result
		entropies_df.loc[name, :] = entropy_result

		# Compute the tb l2 norm for occupancy and entropy for each run
		l2_occ = l_norm_t_b(config1, occupancy_result)
		l2_entropy = l_norm_t_b(config1, entropy_result)

		l2s_df.loc[name, 'entropy_tb_l2'] = l2_entropy
		l2s_df.loc[name, 'occupancy_tb_l2'] = l2_occ

	return occupancy_df, entropies_df, l2s_df


def plot_snr(group_mean_l2s_df, key):
	
	intergenic_group_vals = group_mean_l2s_df.loc['intergenic']
	gene_group_vals = group_mean_l2s_df.loc['gene']

	plt.figure(figsize=(9, 3))
	plt.subplot(1, 2, 1)
	plt.plot(gene_group_vals.index, gene_group_vals[key])
	plt.plot(intergenic_group_vals.index, intergenic_group_vals[key])
	plt.xscale('log')

	eps = 0#np.quantile(group_mean_l2s_df, q=0.1)
	snr = (gene_group_vals[key]+eps) - (intergenic_group_vals[key]+eps)
	plt.subplot(1, 2, 2)
	plt.plot(snr)
	plt.xscale('log')
	
	idx_max = snr.argmax()
	return (idx_max, snr.index[idx_max], eps)


def plot_top_bottom_curves(config1, kappa, meta_df, entropies_df):

	t_indices = config1.get_Hpositions_for_branch('t')
	b_indices = config1.get_Hpositions_for_branch('b')

	group_entropies = entropies_df.join(meta_df[['group']]).reset_index()
	group_entropies.kappa = group_entropies.kappa.round(5)
	group_entropies = group_entropies.set_index(['group', 'kappa', 'name'])

	optim_gene_entropy_mean = group_entropies.loc['gene'].groupby('kappa').mean().loc[kappa]
	optim_intergenic_entropy_mean = group_entropies.loc['intergenic']\
		.groupby('kappa').mean().loc[kappa]

	t_tps = config1.get_timepoints_for_branch('t')
	b_tps = config1.get_timepoints_for_branch('b')

	plt.figure(figsize=(9, 5))
	plt.subplot(2, 2, 1)
	plt.plot(t_tps, optim_gene_entropy_mean[t_indices])
	plt.plot(b_tps, optim_gene_entropy_mean[b_indices])
	plt.ylim(4.6, 5.5)

	plt.subplot(2, 2, 2)
	plt.plot(t_tps, optim_intergenic_entropy_mean[t_indices])
	plt.plot(b_tps, optim_intergenic_entropy_mean[b_indices])
	plt.ylim(4.6, 5.5)

	plt.subplot(2, 2, 3)
	plt.plot(t_tps, group_entropies.loc['gene'].loc[kappa][t_indices].T, c='red')
	plt.plot(b_tps, group_entropies.loc['gene'].loc[kappa][b_indices].T, c='blue')
	plt.ylim(4.3, 5.7)

	plt.subplot(2, 2, 4)
	plt.plot(t_tps, group_entropies.loc['intergenic'].loc[kappa][t_indices].T, c='red')
	plt.plot(b_tps, group_entropies.loc['intergenic'].loc[kappa][b_indices].T, c='blue')
	plt.ylim(4.3, 5.7)

def plot_summed_rows(config1, F):

	t_indices = config1.get_Hpositions_for_branch('t')
	b_indices = config1.get_Hpositions_for_branch('b')

	F_imgs = F.reshape((149, 26, -1))
	F_collapsed_rows = F_imgs.mean(axis=1)

	plt.figure(figsize=(11, 2))
	plt.subplot(1, 3, 1)
	plt.imshow(F_collapsed_rows[t_indices], aspect='auto', cmap='magma_r', vmin=0, vmax=5)

	plt.subplot(1, 3, 2)
	plt.imshow(F_collapsed_rows[b_indices], aspect='auto', cmap='magma_r', vmin=0, vmax=5)

	plt.subplot(1, 3, 3)
	plt.imshow(F_collapsed_rows[b_indices]-F_collapsed_rows[t_indices], 
		aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
