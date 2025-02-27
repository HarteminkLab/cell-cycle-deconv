

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

	from src.geneset import get_deconvolved_geneset
	from src.RealDataReplication import read_no_copy_correction_n_fr_b

	genes = get_deconvolved_geneset()
	orfname = get_orfname(gene_name)

	gene = genes.loc[orfname]
	tss = gene.TSS
	chrom = gene.chr
	padding_2 = window//2
	mnase_span = tss-padding_2, tss+padding_2

	try:
		_, N, frep, b = read_n_fr_b(chrom, mnase_span)
	except KeyError:
		print(f"No replication data for chr{chrom}, {mnase_span}. Reverting to no correction.")
		N, frep, b = read_no_copy_correction_n_fr_b(config1.H)
	
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


def deconvolve_and_plot_region(config, chrom, mnase_span, gamma, kappa, 
		f_savepath=None, save_plots=True):

	from src.RealDataReplication import read_no_copy_correction_n_fr_b

	try:
		_, N, frep, b = read_n_fr_b(chrom, mnase_span)
	except KeyError:
		print(f"No replication data for chr{chrom}, {mnase_span}. Reverting to no correction.")
		N, frep, b = read_no_copy_correction_n_fr_b(config.H)
	
	chromatin_model = ChromatinModel(config)
	chromatin_model.load_mnase_span(chrom=chrom, mnase_span=mnase_span, log=False)

	G_values = chromatin_model.G
	
	chromatin_solver = ChromatinDeconvolveSolver(config, 
		G_values, N, b, frep)
	full_deconvolved_F = chromatin_solver.deconvolve_G_iteratively(gamma=gamma, 
		kappa=kappa, verbose=False)

	if f_savepath is not None:
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

		full_path = row.save_path

		# Compute the entropy and occupancy scores for each of the runs
		try:
			occupancy_result, entropy_result = compute_occupancy_entropy_for_file(full_path)
		except FileNotFoundError:
			continue

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
	kappa_values = gene_group_vals.index

	plt.figure(figsize=(11, 3))
	plt.subplot(1, 2, 1)
	plt.plot(kappa_values, gene_group_vals[key], lw=2, 
		label="Daughter-specific")
	plt.plot(kappa_values, intergenic_group_vals[key], lw=2,
		label="Intergenic")
	plt.xscale('log')
	plt.legend()
	plt.xlabel("$\\kappa$")
	plt.xlim(kappa_values[0], kappa_values[-1])
	plt.title("L2 norm of [DG1 - CG1]")

	eps = 1e-3
	snr = (gene_group_vals[key]+eps) / (intergenic_group_vals[key]+eps)

	plt.subplot(1, 2, 2)
	plt.plot(snr, lw=2, color='black')
	plt.xscale('log')
	plt.title("Signal-to-noise ratio (Daughter-specific / Intergenic)")
	plt.xlabel("$\\kappa$")
	
	offset = 0

	print(snr)
	idx_max = snr.iloc[offset:].argmax()+offset

	# Plot optimal and neighbors for comparison
	indices = [idx_max]#[idx_max-5, idx_max, idx_max+4]

	import matplotlib.patheffects as patheffects

	for i in indices:
		plt.axvline(kappa_values[i], c='red', alpha=1)
		plt.text(kappa_values[i], 1.7, f"{kappa_values[i]:.5f}", color='red', alpha=1.0,
			ha='center', fontsize=9, path_effects=[patheffects.withStroke(linewidth=2, 
				foreground='white')])

	plt.xlim(kappa_values[0], kappa_values[-1])

	plt.suptitle("Selection of optimal L2 regularization weight, $\\kappa$", fontsize=16)
	plt.subplots_adjust(top=0.77)

	return (idx_max, snr.index[idx_max], eps)


def plot_top_bottom_curves(config1, kappa, meta_df, entropies_df):

	from src.expression_chromatin_plots import draw_phase_label_annotations

	entropies_df = entropies_df.copy()
	entropies_df['name'] = meta_df['name']
	entropies_df['kappa'] = meta_df['kappa']
	entropies_df['group'] = meta_df['group']

	t_indices = config1.get_Hpositions_for_branch('t')
	b_indices = config1.get_Hpositions_for_branch('b')

	group_entropies = entropies_df
	group_entropies.kappa = group_entropies.kappa.round(5)
	group_entropies = group_entropies.set_index(['group', 'kappa', 'name'])


	t_tps = config1.get_timepoints_for_branch('t')
	b_tps = config1.get_timepoints_for_branch('b')
	b_tps = t_tps

	fig = plt.figure(figsize=(7, 9))

	num_examples = 4
	plt.subplot(1+num_examples, 2, 1)

	normed = (group_entropies.values - group_entropies.mean(axis=1).values[:, None])
	normed_df = group_entropies.copy()
	normed_df.loc[:] = normed

	optim_gene_entropy_mean = normed_df.loc['gene'].groupby('kappa').mean().loc[kappa]
	optim_intergenic_entropy_mean = normed_df.loc['intergenic']\
		.groupby('kappa').mean().loc[kappa]

	plt.axhline(0, c='black', lw=0.5, ls='solid')

	plt.plot(t_tps, normed_df.loc['gene'].loc[kappa][t_indices].T, c='red', alpha=0.15)
	plt.plot(b_tps, normed_df.loc['gene'].loc[kappa][b_indices].T, c='blue', alpha=0.15)

	plt.plot(t_tps, optim_gene_entropy_mean[t_indices], c='red', label='Mother (mean)', lw=3)
	plt.plot(b_tps, optim_gene_entropy_mean[b_indices], c='blue', label='Daughter (mean)', lw=3)
	# plt.ylim(4, 5.7)

	plt.title("Daughter-specific genes")
	plt.xticks([])
	plt.ylabel("Mean normalized\nnucleosome entropy")
	# plt.legend(ncol=2)
	plt.xlim(b_tps[0], b_tps[-1])
	draw_phase_label_annotations(plt.gca(), config1, flip=True, 
		annotations_x=-0.4, phases=['CG1', 'postG1'], 
		phase_names=['CG1/DG1', 'S,G2,M'])
	plt.ylim(-0.5, 0.5)

	plt.subplot(5, 2, 2)
	plt.axhline(0, c='black', lw=0.5, ls='solid')
	plt.plot(t_tps, optim_intergenic_entropy_mean[t_indices], c='red', label='Mother (mean)', lw=3)
	plt.plot(b_tps, optim_intergenic_entropy_mean[b_indices], c='blue', label='Daughter (mean)', lw=4)
	plt.plot(t_tps, normed_df.loc['intergenic'].loc[kappa][t_indices].T, c='red',  alpha=0.15)
	plt.plot(b_tps, normed_df.loc['intergenic'].loc[kappa][b_indices].T, c='blue', alpha=0.15)
	plt.title("Intergenic regions")
	plt.xticks([])
	plt.xlim(b_tps[0], b_tps[-1])
	plt.yticks([])
	plt.ylim(-0.5, 0.5)

	draw_phase_label_annotations(plt.gca(), config1, flip=True, 
		annotations_x=-0.4, phases=['CG1', 'postG1'], 
		phase_names=['CG1/G1', 'S,G2,M'])

	for i in range(num_examples):
		plt.subplot(5, 2, 3+i*2)
		plt.axhline(0, c='black', lw=0.5, ls='solid')
		plt.plot(t_tps, normed_df.loc['gene'].loc[kappa].iloc[i][t_indices], c='red')
		plt.plot(b_tps, normed_df.loc['gene'].loc[kappa].iloc[i][b_indices], c='blue')
		plt.ylim(-0.5, 0.5)
		plt.xticks([])
		plt.xlim(b_tps[0], b_tps[-1])
		plt.title(normed_df.loc['gene'].loc[kappa].iloc[i].name)

		plt.subplot(5, 2, 4+i*2)
		plt.axhline(0, c='black', lw=0.5, ls='solid')
		plt.plot(t_tps, normed_df.loc['intergenic'].loc[kappa].iloc[i][t_indices], c='red')
		plt.plot(b_tps, normed_df.loc['intergenic'].loc[kappa].iloc[i][b_indices], c='blue')
		plt.ylim(-0.5, 0.5)
		plt.yticks([])
		plt.xlim(b_tps[0], b_tps[-1])
		plt.xticks([])
		plt.title(normed_df.loc['intergenic'].loc[kappa].iloc[i].name)
	

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


def load_intergenic_chromatin_F_from_sweep_run(config1, meta_df, intergenic_name, kappa, 
	window):
	""""Load the chromatin deconvolution data from the kappa sweep run
	for an intergenic region by name"""

	win_2 = window//2
	intergenic_name_spl = intergenic_name.split('_')
	chrom = int(intergenic_name_spl[0])
	mid = int(intergenic_name_spl[1])
	mnase_span = mid-win_2, mid+win_2

	selected_meta_rows_df = meta_df.loc[intergenic_name].reset_index()
	selected_meta_rows_df.kappa = selected_meta_rows_df.kappa.round(5)
	selected_meta_rows_df = selected_meta_rows_df.set_index('kappa')
	filepath = selected_meta_rows_df.loc[kappa].path

	chromatin_F = np.load(filepath)
	F_imgs = chromatin_F.reshape((config1.H.shape[1], 26, -1))
	return F_imgs, chrom, mnase_span


def load_gene_chromatin_F_from_sweep_run(config1, meta_df, gene_name, kappa, window):
	""""Load the chromatin deconvolution data from the kappa sweep run
	for a gene"""
	from src.sgd import get_orfname
	from src.sgd import read_nondubious_genes_dataset

	orfname = get_orfname(gene_name)
	genes = read_nondubious_genes_dataset()

	win_2 = window//2
	gene = genes.loc[orfname]
	tss = gene.TSS
	chrom = gene.chr
	mnase_span = tss-win_2, tss+win_2

	selected_row = meta_df.loc[gene_name]
	#
	# todo: currently a single kappa,
	#       need to reimplement with consistent data frame key

	#selected_meta_rows_df.kappa = selected_meta_rows_df.kappa.round(5)
	#selected_meta_rows_df = selected_meta_rows_df.set_index('kappa')
	filepath = selected_row.path

	chromatin_F = np.load(filepath)
	F_imgs = chromatin_F.reshape((config1.H.shape[1], 26, -1))
	return F_imgs, chrom, mnase_span

