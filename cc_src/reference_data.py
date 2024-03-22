

import pandas as pd


def load_spellman_orfs():
	"""Load the spellman orfs list from the xin gene association table"""
	gene_as = pd.read_csv('datasets/datasets_from_web_deconvolution.cs.duke.edu/gene_associated.tsv', sep='\t')
	gene_as = gene_as[gene_as['Spellman1998'] == '1']
	spellman_orfs = gene_as['Systematic Name'].values
	return spellman_orfs


def load_xin_1500_cc_orfs():
	"""The cell cycle genes that Xin has annotated represents:

	Deconvolved PTR score threshold corresponding to the 1,500 most strongly cell-cycle–regulated genes.

	The gene association table does not have a label, but they are sorted by the deconvolved PTR scores.
	So take that highest 1500 genes.
	"""
	gene_as = pd.read_csv('datasets/datasets_from_web_deconvolution.cs.duke.edu/gene_associated.tsv', sep='\t')
	xin_cell_cycle_genes = gene_as.iloc[0:1500]['Systematic Name'].values
	return xin_cell_cycle_genes


def load_analysis_genes():
	# Read from the appropriate csv file to load the genes we will include in
	# our analysis, filtered for coverage, any other criteria we may want to 
	# add later

	# geneset_cov90 = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies_cutoff_cov90.csv').set_index('orf_name')
	geneset_depth = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies_cutoff_depth170.csv').set_index('orf_name')
	return geneset_depth


def load_plus_ones(replicate=1):
	return pd.read_csv(f"datasets/computed_mnase/rep{replicate}_plus_ones.csv").set_index('orf_name')


def read_macisaac_sites():

	from cc_src.read_bam import _fromRoman

	sites = pd.read_csv('data/reference_data/p005_c2.sacCer3.gff.txt', sep='\t',
			   names=range(9))
	sites = sites[sites.columns[[0, 3, 4, 6, 8]]].copy()
	sites.columns = ['chr','start','stop','strand','TF']

	#sites.chr = _fromRoman(sites.chr)
	sites.TF = sites.TF.str.replace(';', '').str.replace('Site ', '')
	return sites
