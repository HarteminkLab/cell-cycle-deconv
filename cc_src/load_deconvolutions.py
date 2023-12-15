
from src.deconvolution_model import Model
# File to make loading the model and deconvolutions quicker
# TODO: Rethink pathing of the deconvolution models, etc...
# combine cell cycle and deconvolution repositories..


def load_yl_rep2_gene_body_nucleosome_entropy_deconvolution():
	model_path = ('../deconvolution-project/saved_output/2023-10-17'\
				'_yl_rep2_entropy_nuc_genebody_rg1_sigma0_11/wt2_rg1_test_sigma0.label')
	parentdir = ('../deconvolution-project/saved_output/'\
			  '2023-10-17_yl_rep2_entropy_nuc_genebody_rg1_sigma0_11/')
	configpath = ('../deconvolution-project/saved_output/2023-10-17_'\
				'yl_rep2_entropy_nuc_genebody_rg1_sigma0_11/'\
				'DeconvYLRep2GeneBodyNucEntropies.matlab')

	gb_ne_model = Model(model_path, parentdir, configpath)
	gb_nuc_entropy_ptrs = gb_ne_model.calculate_ptrs()
	return gb_ne_model, gb_nuc_entropy_ptrs


def load_yl_rep2_small_fragment_occupancy_deconvolution():
	model_path = ('../deconvolution-project/saved_output/2023-10-06_yl_rep2_chrom_small_prom_rg1_sigma0_11/'\
				  'wt2_rg1_test_sigma0.label')
	parentdir = '../deconvolution-project/saved_output/2023-10-06_yl_rep2_chrom_small_prom_rg1_sigma0_11/'
	configpath = ('../deconvolution-project/saved_output/2023-10-06_yl_rep2_chrom_small_prom_rg1_sigma0_11/'\
				  'DeconvYLRep2ChromSmallProm.matlab')

	model = Model(model_path, parentdir, configpath)
	sm_prom_ptrs = model.calculate_ptrs()
	return model, sm_prom_ptrs

def load_yl_rep2_gene_expression_deconvolution():
	model_path = '../deconvolution-project/saved_output/2023-09-26_yl_rep2_all_genes_rg1/wt2_rg1_test_sigma0.label'
	parentdir = '../deconvolution-project/saved_output/2023-09-26_yl_rep2_all_genes_rg1/'
	configpath = '../deconvolution-project/saved_output/2023-09-26_yl_rep2_all_genes_rg1/DeconvYLRep2Sigma011.m'

	model_ge = Model(model_path, parentdir, configpath)
	ge_ptrs = model_ge.calculate_ptrs()
	return model_ge, ge_ptrs
