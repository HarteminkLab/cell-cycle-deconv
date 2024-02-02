
from src.model import Model
from src.dynamic_config_alpha import create_dynamic_alpha_config
import numpy as np
import pandas as pd


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



def search_alphas(gene_name, posteriors_filepath, alphas):

	dg1_props = []

	for alpha in alphas:

		config = create_dynamic_alpha_config(posteriors_filepath, alpha, 1, "Dynamic config")

		model1 = Model(config, gene_name, 0.0)
		model1.deconvolve_find_optimal_gamma()

		dg1_prop = compute_prop_in_dg1(model1)
		dg1_props.append(dg1_prop)
		
	ret_df = pd.DataFrame({"alpha": alphas, "prop_dg1": dg1_props})
	return ret_df


from src.timer import Timer

def perform_alpha_search_gene(gene_name, alphas = np.arange(0, 40, 1), replicate=1):
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
    
    timer = Timer()
    print(f"Computing {len(alphas)} alpha values for {gene_name}...", end="")
    dg1_df = search_alphas(gene_name, posteriors_filepath, alphas)
    print(f"Done in {timer.get_time()}")

    return dg1_df
