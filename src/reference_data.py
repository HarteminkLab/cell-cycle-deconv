

import pandas as pd
import matplotlib.pyplot as plt


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

	orfs_w_h3k56ac


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


def load_chrom_replication_timing():

	from src.config import load_configs_by_config_type

	replication_timing = pd.read_csv(
		'data/replication_timing/yl_2019/chrom_replication_timing_shared.csv')
	replication_timing = replication_timing.set_index(['chr', 'start'])
	replication_timing['replication_time_1'] = 0
	replication_timing['replication_time_2'] = 0

	config1, config2 = load_configs_by_config_type('shared')

	for index, row in replication_timing.iterrows():
		repl_index = row.replication_index
		replication_tp_repl1 = config1.get_timepoint_for_index(repl_index)
		replication_tp_repl2 = config2.get_timepoint_for_index(repl_index)
		replication_timing.loc[index, 'replication_time_1'] = replication_tp_repl1
		replication_timing.loc[index, 'replication_time_2'] = replication_tp_repl2

	replication_timing['replication_time'] = (replication_timing.replication_time_1+replication_timing.replication_time_2)/2

	return replication_timing


def plot_guo_gene_expression(orf_name):
	guo_f_df = pd.read_csv('datasets/datasets_from_web_deconvolution.cs.duke.edu/deconvolved_profiles.tsv', 
		sep='\t').set_index('SystematicName')
	tp_cols = guo_f_df.columns[2:]

	gene = guo_f_df.loc[orf_name][tp_cols]

	df_index = tp_cols

	r_values = [x for x in df_index if x.startswith('R')]
	d_values = [x for x in df_index if x.startswith('D')]
	c_values = [x for x in df_index if x.startswith('C')]

	# To get just the numbers for each:
	r_numbers = [int(x.split(':')[1].rstrip(')')) for x in r_values]
	d_numbers = [int(x.split(':')[1].rstrip(')')) for x in d_values]
	c_numbers = [int(x.split(':')[1].rstrip(')')) for x in c_values]

	plt.figure(figsize=(9, 2))
	plt.subplot(1, 3, 1)
	plt.plot(r_numbers, gene[r_values])
	plt.title("Recovery")
	plt.ylim(0, gene.max()*1.1)

	plt.subplot(1, 3, 2)
	plt.plot(c_numbers, gene[c_values])
	plt.title("Mother")
	plt.ylim(0, gene.max()*1.1)

	plt.subplot(1, 3, 3)
	plt.plot(d_numbers, gene[d_values])
	plt.title("Daughter")
	plt.ylim(0, gene.max()*1.1)


def load_plus_ones():
	rep1_p1 = pd.read_csv('datasets/computed_mnase/rep1_plus_ones.csv').set_index('orf_name')
	rep2_p1 = pd.read_csv('datasets/computed_mnase/rep2_plus_ones.csv').set_index('orf_name')
	plus_ones_combined = rep1_p1.join(rep2_p1, lsuffix='_rep1', rsuffix='_rep2')
	plus_ones_combined['combined_+1'] = (plus_ones_combined['+1_rep1']+plus_ones_combined['+1_rep2'])//2.
	return plus_ones_combined.dropna()


def load_p1_gene_regions():
	"""We have plus ones called, we can better identify promoters and gene bodies.
	In this case we will include the +1 and define the gene body as 500 bps.

	The promoter is defined as 380 upstream of the called plus one, to -80 of of the plus one. (to omit
	the plus nucleosome in the promoter.)
	"""

	from src.reference_data import load_plus_ones
	from src.geneset import get_deconvolved_geneset

	genes = get_deconvolved_geneset()
	plus_one_locations = load_plus_ones()

	# Create a dataframe that defines the windows we are interested in computing

	# Span relative to called +1 position (assuming watson strand)
	promoter_span = (-380, -80) # Approximately 300 bp promoter span, exclusive of +1
	gene_body_span = (-80, 420) # Inclusion of +1, +2, +3 nucleosomes

	gene_metric_boundaries = genes.join(plus_one_locations[['combined_+1']])\
	    [['gene', 'strand', 'length', 'chr', 'combined_+1', ]]

	is_watson = gene_metric_boundaries.strand == '+'
	is_crick = gene_metric_boundaries.strand == '-'

	watson_p1s = gene_metric_boundaries.loc[is_watson, 'combined_+1']
	crick_p1s = gene_metric_boundaries.loc[is_crick, 'combined_+1']

	# Watson promoters
	gene_metric_boundaries.loc[is_watson, 'promoter_start'] = watson_p1s+promoter_span[0]
	gene_metric_boundaries.loc[is_watson, 'promoter_end'] = watson_p1s+promoter_span[1]

	# Crick promoters
	gene_metric_boundaries.loc[is_crick, 'promoter_start'] = crick_p1s-promoter_span[1]
	gene_metric_boundaries.loc[is_crick, 'promoter_end'] = crick_p1s-promoter_span[0]

	# Watson Gene Bodies
	gene_metric_boundaries.loc[is_watson, 'gene_body_start'] = watson_p1s+gene_body_span[0]
	gene_metric_boundaries.loc[is_watson, 'gene_body_end'] = watson_p1s+gene_body_span[1]

	# Crick Gene Bodies
	gene_metric_boundaries.loc[is_crick, 'gene_body_start'] = crick_p1s-gene_body_span[1]
	gene_metric_boundaries.loc[is_crick, 'gene_body_end'] = crick_p1s-gene_body_span[0]

	return gene_metric_boundaries
