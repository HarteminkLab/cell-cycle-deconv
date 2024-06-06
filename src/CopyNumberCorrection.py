
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.mnase_replication_timing_analysis import get_bin_for_position
from src.geneset import get_deconvolved_geneset
from src.helpers import calcH_config
from src.delta_config import load_yl_delta_config
from src.config import read_yl_vst_data_rep


class CopyNumberCorrector:

	def __init__(self):
		# Load the gene expression data and geneset
		self.ge_data = read_yl_vst_data_rep(1)
		self.genes = get_deconvolved_geneset()
		self.normalized_ge = normalize_total_reads(self.ge_data)
		

	def correct_for_copy_number(self):
		from src.timer import Timer
		from src.helpers import calcH_config

		genes = self.genes_w_repl_timing
		repl_profile = pd.read_csv('datasets/computed_mnase/'\
		    'deconvolved_all_chr_replication_profile_delta_model.csv')
		repl_profile = repl_profile.set_index(['chr', 'start'])

		config = load_yl_delta_config(1)
		H, Hpos = calcH_config(config)
		    
		timer = Timer()
		i = 0
		corrected_gene_expression = self.normalized_ge.copy()

		for orf_name, gene in genes.iterrows():
		    
		    if i % 1000 == 0:
		        timer.print_time(f"{i}/{len(genes)}")

		    gene_expression = self.normalized_ge.loc[gene.name]

		    corrected_data = copy_number_correct(H, gene, gene_expression, 
		        repl_profile, genes)
		    corrected_gene_expression.loc[orf_name] = corrected_data

		    i += 1

		self.corrected_gene_expression = corrected_gene_expression
		self.normalized_corrected_ge = normalize_total_reads(corrected_gene_expression)

	def compute_replication_timing(self):
		from src.timer import Timer
		from src.helpers import calcH_config

		genes = self.genes
		repl_profile = pd.read_csv('datasets/computed_mnase/'\
		    'deconvolved_all_chr_replication_profile_delta_model.csv')
		repl_profile = repl_profile.set_index(['chr', 'start'])

		config = load_yl_delta_config(1)
		H, Hpos = calcH_config(config)
		    
		repl_timing = self.genes[[]].copy()
		repl_timing['replication_time'] = np.nan
		repl_timing['replication_H_index'] = np.nan

		for orf_name, gene in genes.iterrows():

			replication_profile = get_gene_replication_profile(gene.name, repl_profile, self.genes)
			replication_idx = replication_profile.round().argmax()

			if replication_idx >= 0:
				repl_timing.loc[orf_name, 'replication_H_index'] = replication_idx

			if replication_idx >= 0:
				repl_timing.loc[orf_name, 'replication_time'] = config.get_timepoint_for_index(replication_idx)

		self.genes_w_repl_timing = self.genes.join(repl_timing).dropna()


	def compute_ptrs(self):
		from src.peak_to_trough import compute_quantile_ptr

		ge_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		    0.2, 0.8), 1, self.normalized_ge.values)
		corrected_ge_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.normalized_corrected_ge.values)

		ge_comparison_ptr_df = pd.DataFrame({
		    "raw_ptr": ge_ptrs,
		    "corrected_ptr": corrected_ge_ptrs,
		}, index=self.ge_data.index)

		# Select only genes in our gene set (filtered for low read count)
		self.ge_comparison_ptr_df = ge_comparison_ptr_df.loc[self.genes_w_repl_timing.index]

	def plot_ptrs(self, orfs=None):
		plt.figure(figsize=(4, 4))

		dat = self.ge_comparison_ptr_df

		if orfs is not None:
			dat = dat.loc[orfs]

		dat = dat.join(self.genes_w_repl_timing[['replication_time']], how='inner')
		dat = dat.sort_values('replication_time', ascending=True)

		plt.scatter(dat.raw_ptr, dat.corrected_ptr, s=1,
			c=dat.replication_time, cmap='viridis_r')
		plt.title(f"Uncorrected PTR vs Corrected PTR values,\nn={len(dat)}")
		plt.xlabel("Raw expression PTR")
		plt.ylabel("Copy-number-corrected expression PTR")


	def plot_corrected_gene(self, orf_name):
		gene = self.genes.loc[orf_name]
		plt.figure(figsize=(4, 3))
		plt.plot(self.normalized_ge.loc[gene.name])
		plt.plot(self.normalized_corrected_ge.loc[gene.name], ls='dotted')
		plt.title(orf_name)

def normalize_total_reads(ge_data, total_counts=50000):
    normalized_reads = ge_data / \
        ge_data.sum(axis=0).values.reshape((1, -1)) * total_counts
    return normalized_reads


def get_gene_replication_profile(orf_name, repl_profile=None, geneset=None):
	"""Get the replication profile for a given orf. Note the profile returned
	using the Delta-DG1 model.
	
	Loads the deconvolved profile for the given gene orf, from the gene's 
	assigned bin
	
	Returns vector of the deconvolved profile [1 and 2's representing the copy number
	at each timepoint]
	"""

	if repl_profile is None:
		repl_profile = pd.read_csv('datasets/computed_mnase/'\
			'deconvolved_all_chr_replication_profile_delta_model.csv')
		repl_profile = repl_profile.set_index(['chr', 'start'])

	if geneset is None:
		from src.geneset import get_deconvolved_geneset
		geneset = get_deconvolved_geneset()

	gene = geneset.loc[orf_name]
	chrom = gene.chr
	start_indices = repl_profile.loc[chrom].index.values

	chrom_repl_profile = repl_profile.loc[chrom]
	bin_idx, bin_start_bp = get_bin_for_position(gene.TSS, start_indices)
	gene_repl_profil = chrom_repl_profile.loc[bin_start_bp]
	gene_repl_profil.index = gene_repl_profil.index.astype(int)
	
	return gene_repl_profil
	

def copy_number_correct(H, gene, data_to_correct, repl_profiles, geneset):
	"""Correct the gene data by copy number. Uses the H matrix and the replication timing
	for a gene to effectly reduce the proportion of data in copy number 2 timepoints to 
	1. 
	"""

	# Identify the time of replication
	replication_profile = get_gene_replication_profile(gene.name, repl_profiles, geneset)
	replication_idx = replication_profile.round().argmax()
	
	# Within the H matrix, identify the subset of
	# postG1 that will be copy number 2
	copy_2_H = H[:, replication_idx:-1]

	# Compute the proportion of copy number 2
	# and copy number 1 cells
	prop_cop2 = copy_2_H.sum(axis=1)
	prop_cop1 = 1 - prop_cop2
	
	# Correct the data by halving
	# the copy number 1 proportion
	data_cop1 = data_to_correct * prop_cop1
	data_cop2 = data_to_correct * prop_cop2 * 0.5
	
	# Recombine the copy number 1 and copy number 2 (corrected)
	data_corrected = data_cop1 + data_cop2

	return data_corrected
	
