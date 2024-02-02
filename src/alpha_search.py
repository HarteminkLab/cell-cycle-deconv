
from src.model import Model
from src.dynamic_config_alpha import create_dynamic_alpha_config
import numpy as np
import pandas as pd
from src.timer import Timer


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

	for alpha in alphas:

		config = create_dynamic_alpha_config(posteriors_filepath, alpha, replicate, "Dynamic config")

		model1 = Model(config, gene_name, 0.0)
		model1.deconvolve_find_optimal_gamma()

		dg1_prop = compute_prop_in_dg1(model1)
		dg1_props.append(dg1_prop)
		
	ret_df = pd.DataFrame({"alpha": alphas, "prop_dg1": dg1_props})
	return ret_df


def perform_alpha_search_gene(gene_name, alphas = np.arange(0, 40, 1), replicate=None, timer=None):
	"""
	Performs the DG1 proportion calculation for each alpha value and returns
	a dataframe of the results
	"""
	
	if replicate == 1:
		posteriors_filepath = 'data/yl_2019_replicate1/posteriors.txt'
	elif replicate == 2:
		posteriors_filepath = 'data/yl_2019_replicate2/posteriors.txt'
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

	genes = ["DSE1", "DSE2", "DSE3", "DSE4"]
	min_a, max_a, step_a = 0, 50, 1
	alpha_values = np.arange(min_a, max_a, step_a)

	# Perform alpha search for replicate 1
	dg1_rep1_all_genes_df = pd.DataFrame()
	for gene in genes:
	    dg1_df = perform_alpha_search_gene(gene, alphas=alpha_values, timer=timer, replicate=1)
	    dg1_df['gene'] = gene
	    dg1_rep1_all_genes_df = pd.concat([dg1_rep1_all_genes_df, dg1_df])

	# Perform alpha search for replicate 2
	dg1_rep2_all_genes_df = pd.DataFrame()
	for gene in genes:
	    dg1_df = perform_alpha_search_gene(gene, alphas=alpha_values, timer=timer, replicate=2)
	    dg1_df['gene'] = gene
	    dg1_rep2_all_genes_df = pd.concat([dg1_rep2_all_genes_df, dg1_df])

	# Combine results and save to disk
	dg1_rep1_all_genes_df['replicate'] = 1
	dg1_rep2_all_genes_df['replicate'] = 2
	combined_dg1_df = pd.concat([dg1_rep1_all_genes_df, dg1_rep2_all_genes_df])

	save_file = f'output/dg1_alpha_search_{min_a}_{max_a}_{step_a}.csv'
	combined_dg1_df.to_csv(save_file)

	print(f"Save to: {save_file}")

	return combined_dg1_df



if __name__ == '__main__':	
	main()
