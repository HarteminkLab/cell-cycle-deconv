

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

	geneset_depth = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies_cutoff_depth170.csv').set_index('orf_name')
	return geneset_depth


def load_plus_ones(replicate=1):
	return pd.read_csv(f"datasets/computed_mnase/rep{replicate}_plus_ones.csv").set_index('orf_name')


def read_macisaac_sites():

	from src.read_bam import _fromRoman

	sites = pd.read_csv('data/reference_data/p005_c2.sacCer3.gff.txt', sep='\t',
			   names=range(9))
	sites = sites[sites.columns[[0, 3, 4, 6, 8]]].copy()
	sites.columns = ['chr','start','stop','strand','TF']

	#sites.chr = _fromRoman(sites.chr)
	sites.TF = sites.TF.str.replace(';', '').str.replace('Site ', '')
	return sites

def read_rossi_sites():

	import os
	# let's read in the chip-exo rossi bed files
	def read_filename_chip_exo(filename):
		
		if 'filtered_new' in filename:
			tf_name = filename.split('/')[-1].split('_')[1]
		else:
			tf_name = filename.split('/')[-1].split('_')[0]

		chip_df = pd.read_csv(filename, sep='\t', header=None)
		chip_df.columns = ['chr', 'start', 'stop', 'name', '?', '.']
		chip_df['tf'] = tf_name
		return chip_df


	rossi_dir = 'datasets/chip_exo_rossi/04_ChExMix_Peaks/'
	bedfiles = os.listdir(rossi_dir)
	fullpaths = [f"{rossi_dir}{f}" for f in bedfiles]

	rossi_sites = pd.DataFrame()
	for filepath in fullpaths:
		if filepath.endswith('.bed'):
			tf_sites = read_filename_chip_exo(filepath)
			rossi_sites = pd.concat([rossi_sites, tf_sites])

	rossi_sites.chr = rossi_sites.chr.str.replace('chr', '').astype(int)
	rossi_sites = rossi_sites.reset_index(drop=True)
	rossi_sites = rossi_sites.sort_values(['chr', 'start'])

	return rossi_sites

def load_h3k56ac_marks(full=False):
	"""Load histone marks"""

	histone_mods_df = pd.read_csv('/Users/trung/Research/_archive/data/reference_data/molcel_5341_mmc4',
		index_col=0, skiprows=[1])

	# Get the first column (steady state histone acetylation value)
	h3k56_acetylation = histone_mods_df[['H3K56ac']]
	
	# Nucleosomes associated with histone marks, with associated ORFs if needed
	histone_mod_nuc_db = pd.read_csv('/Users/trung/Research/_archive/data/reference_data/mmc3.csv')
	histone_mod_nuc_db = histone_mod_nuc_db.set_index('nuc_id')
	histone_mod_nuc_db.sort_values(['chr', 'center'])

	orfs_w_h3k56ac = h3k56_acetylation.join(histone_mod_nuc_db, how='inner')
	orfs_w_h3k56ac = orfs_w_h3k56ac[['H3K56ac', 'acc', 'gene']]

	return orfs_w_h3k56ac


def save_PAS_nucs_to_disk():
	from glob import glob

	meta_paths = glob('output/deconvolve_sharedg1_g0066_11x11_PAS_2024_09_12/chromatin/*meta*')
	meta_df = pd.DataFrame()
	for meta_path in meta_paths:
	    loaded_dat = pd.read_csv(meta_path).set_index('Unnamed: 0')
	    meta_df = pd.concat([meta_df, loaded_dat])
	meta_df['replicate_1_PAS_nuc'] = meta_df['rep1_+1']
	meta_df['replicate_2_PAS_nuc'] = meta_df['rep1_+2']
	meta_df = meta_df[['replicate_1_PAS_nuc', 'replicate_2_PAS_nuc']].reset_index().rename(columns={
	    'Unnamed: 0': 'orf_name'}).set_index('orf_name')
	meta_df.to_csv('datasets/computed_mnase/computed_PAS_nucs.csv')
