
from src.model import Model
from src.dynamic_config_alpha import create_dynamic_alpha_config
import numpy as np
import pandas as pd
from src.timer import Timer
import matplotlib.pyplot as plt


def compute_prop_in_dg1(model1):
	config = model1.config
	
	dg1_indices = config.get_timepoints_phases_Hpositions_for_branch('b')[0][2]
	postg1_indices = config.get_timepoints_phases_Hpositions_for_branch('b')[1][2]

	cg1_indices = config.get_timepoints_phases_Hpositions_for_branch('t')[0][2]
	rg1_indices = config.get_timepoints_phases_Hpositions_for_branch('i')[0][2]

	f = model1.f

	dg1_f = f[dg1_indices]
	cg1_f = f[cg1_indices]
	rg1_f = f[rg1_indices]
	postg1_f = f[postg1_indices]

	return dg1_f.sum() / f.sum()


def search_alphas(gene_name, posteriors_filepath, alphas, replicate):
	"""
	Compute the proportion of expression in DG1/(all expression) for a gene given a set of alpha values
	to search through and the replicate

	TODO: Refactor such that the posteriors are loaded via the replicate parameter
	"""

	dg1_props = []
	fs = []
	gammas = []
	sns = []
	rns = []

	for alpha in alphas:

		config = create_dynamic_alpha_config(posteriors_filepath, alpha, replicate, "Dynamic config")

		model1 = Model(config, gene_name, 0.0)
		model1.deconvolve_find_optimal_gamma()

		dg1_prop = compute_prop_in_dg1(model1)
		dg1_props.append(dg1_prop)

		gammas.append(model1.gamma)
		fs.append(model1.f)
		sns.append(model1.sn)
		rns.append(model1.rn)
		
	ret_df = pd.DataFrame({"alpha": alphas, "prop_dg1": dg1_props, "gamma": gammas, "rn": rns, "sn": sns, "f": fs})

	return ret_df


def compute_peak_dg1s(combined_dg1_df):

	from src.config import load_yl_replicate1_rg1_alpha_vst_config, load_yl_replicate2_rg1_alpha_vst_config
	from src.model import Model

	peaks_df = combined_dg1_df = combined_dg1_df.copy()
	peaks_df = peaks_df.reset_index(drop=True)
	peaks_df['peak_dg1'] = -1

	config1 = load_yl_replicate1_rg1_alpha_vst_config()
	model1 = Model(config1, "CLB2", 0.0)
	dg1_indices = config1.get_timepoints_phases_Hpositions_for_branch('b')[0][2]
	postg1_indices = config1.get_timepoints_phases_Hpositions_for_branch('b')[1][2]
	
	for idx, row in peaks_df.iterrows():
		idx_max = row.f[dg1_indices].argmax()
		peaks_df.loc[idx, 'peak_dg1'] = idx_max

	return peaks_df

	
def plot_DG1_genes_alpha_curves(combined_dg1_df, replicate, alpha, normalize, genes=None):

	from src.config import load_yl_replicate1_rg1_alpha_vst_config, load_yl_replicate2_rg1_alpha_vst_config
	from src.model import Model

	fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3))
	plt.subplots_adjust(wspace=0., top=0.8)

	config1 = load_yl_replicate1_rg1_alpha_vst_config()
	model1 = Model(config1, "CLB2", 0.0)
	cg1_indices = config1.get_timepoints_phases_Hpositions_for_branch('t')[0][2]
	dg1_indices = config1.get_timepoints_phases_Hpositions_for_branch('b')[0][2]
	postg1_indices = config1.get_timepoints_phases_Hpositions_for_branch('b')[1][2]
	gene_rows = combined_dg1_df[(combined_dg1_df.alpha == alpha) & (combined_dg1_df.replicate == replicate)]
	gene_rows = gene_rows.sort_values('peak_dg1')

	i = 0

	if normalize:
		ylim = 0, 15
	else:
		ylim = 5, 35

	for _, row in gene_rows.iterrows():

		if genes is not None:
			if row.gene not in genes:
				continue

		# Normalize f such that min is 0 and max is 10
		f = row.f
		
		if normalize:
			f = f - f.min()
			f = f/f.max() * 10.

		xs_cg1 = np.arange(len(cg1_indices))
		xs_pg1 = np.arange(len(postg1_indices))+xs_cg1.max()
		color = plt.get_cmap('tab10')(i)
		ax0.plot(xs_cg1, f[xs_cg1], color=color, lw=3)
		ax0.plot(np.concatenate([xs_cg1, xs_pg1]), 
			     np.concatenate([f[xs_cg1] , f[postg1_indices]]), color=color, label=row.gene, lw=1)
		ax0.set_ylim(ylim)

		if normalize:
			ax0.set_ylabel("Normalized expression")
		else:
			ax0.set_ylabel("Expression")

		ax0.set_yticks([])
		ax0.set_title("Mother")
		ax0.set_xlim(0, xs_pg1.max())

		xs_dg1 = np.arange(len(dg1_indices))
		xs_pg1 = np.arange(len(postg1_indices))+xs_dg1.max()
		color = plt.get_cmap('tab10')(i)
		ax1.plot(xs_dg1, f[dg1_indices], color=color, lw=3, label=row.gene)
		ax1.plot(np.concatenate([xs_dg1, xs_pg1]), 
			     np.concatenate([f[dg1_indices] , f[postg1_indices]]), color=color, lw=1)
		ax1.set_ylim(ylim)
		ax1.set_yticks([])
		ax1.set_title("Daughter")
		ax1.set_xlim(0, xs_pg1.max())


		i += 1
		
	ax0.legend(ncol=4, fontsize=9)
	plt.suptitle(f"DG1 curves, alpha={alpha}, Replicate={replicate}")
	return fig


def perform_alpha_search_gene(gene_name, alphas = np.arange(0, 40, 3), replicate=None, timer=None):
	"""
	Performs the DG1 proportion calculation for each alpha value and returns
	a dataframe of the results
	"""
	
	if replicate == 1:
		posteriors_filepath = 'data/2019_cloccs_fits/yl_2019_replicate1/posteriors.txt'
	elif replicate == 2:
		posteriors_filepath = 'data/2019_cloccs_fits/yl_2019_replicate2/posteriors.txt'
	else:
		raise ValueError(f"Invalid replicate {replicate}")
	
	if timer is None:
		timer = Timer()

	print(f"Computing {len(alphas)} alpha values for {gene_name}...", end="")
	dg1_df = search_alphas(gene_name, posteriors_filepath, alphas, replicate)
	print(f"Done in {timer.get_time()}")

	return dg1_df


def main():
	"""
	For DSE1-4, compute the proportion of DG1 for both replicates and save the result to a dataframe. 

	Depending on how many alpha values to search through this can take up to an hour. Each alpha value
	can take around 30 seconds, as we are performing a gamma search through for each gene.
	"""

	timer = Timer()

	xin_dg1_genes = ['ASH1','EGT2','AMN1','DSE3','DSE4','PRY3','SCW11','DSE1','DSE2','CTS1']
	dse_genes = ['DSE1','DSE2','DSE3','DSE4']

	min_a, max_a, step_a = 0, 50, 1
	alpha_values = np.arange(min_a, max_a, step_a)

	# Perform alpha search for replicate 1
	dg1_rep1_all_genes_df = pd.DataFrame()
	for gene in dse_genes:
		dg1_df = perform_alpha_search_gene(gene, alphas=alpha_values, timer=timer, replicate=1)
		dg1_df['gene'] = gene
		dg1_rep1_all_genes_df = pd.concat([dg1_rep1_all_genes_df, dg1_df])

	# Perform alpha search for replicate 2
	dg1_rep2_all_genes_df = pd.DataFrame()
	for gene in dse_genes:
		dg1_df = perform_alpha_search_gene(gene, alphas=alpha_values, timer=timer, replicate=2)
		dg1_df['gene'] = gene
		dg1_rep2_all_genes_df = pd.concat([dg1_rep2_all_genes_df, dg1_df])

	# Combine results and save to disk
	dg1_rep1_all_genes_df['replicate'] = 1
	dg1_rep2_all_genes_df['replicate'] = 2
	combined_dg1_df = pd.concat([dg1_rep1_all_genes_df, dg1_rep2_all_genes_df])

	save_file = f'output/alpha_search/dg1_alpha_search_{min_a}_{max_a}_{step_a}.csv'
	combined_dg1_df.to_csv(save_file)

	print(f"Save to: {save_file}")

	return combined_dg1_df



if __name__ == '__main__':	
	main()
