
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.mnase_replication_timing_analysis import get_bin_for_position
from src.geneset import get_deconvolved_geneset
from src.config import read_yl_vst_data_rep, load_configs_by_config_type
from src.stepwise_replication_solver import load_gene_replication_profile

class CopyNumberCorrector:
	"""
	Copy number correction based on old logic.

	todo: May be deprecating this class soon as a newer simpler methodology is being
	created that strictly uses a copy number matrix. Addressing normalization and scaling 
	at the moment.
	"""

	def __init__(self, config_type, replicate):
		
		from src.helpers import calcH_config

		# Load the gene expression data and geneset
		self.genes = get_deconvolved_geneset()
		self.repl_profile = load_gene_replication_profile(config_type)

		self.config1, self.config2 = load_configs_by_config_type(config_type, mode='expression', 
			with_copy_correction=False)
		self.config = self.config1 if replicate == 1 else self.config2

		self.H, Hpos = self.config.calcH_function(self.config.intervals_wt1, self.config.WT1_TIMEPOINTS)
		self.config_type = config_type
		self.replicate = replicate

	def load_gene_expression_data(self):
		self.normalized_reads_data = read_yl_vst_data_rep(self.replicate)

	def correct_for_copy_number(self):
		from src.timer import Timer
		
		H = self.H
		repl_profile = self.repl_profile
			
		timer = Timer()
		i = 0
		corrected_gene_expression = self.normalized_reads_data.copy()

		# For each gene
		for orf_name, gene in repl_profile.iterrows():
			
			if i % 1000 == 0:
				timer.print_time(f"{i}/{len(repl_profile)}")

			# Load the gene expression for the gene
			gene_expression = self.normalized_reads_data.loc[gene.name]

			# Correct the copy number using H and the replication profile
			# and save into new gene expression table
			replication_index = int(repl_profile.loc[gene.name].replication_H_index)
			corrected_data = copy_number_correct_H_index(H, gene_expression, replication_index)
			corrected_gene_expression.loc[orf_name] = corrected_data

			i += 1

		# Then, normalize such that for all genes, the samples
		# are equalized
		self.corrected_gene_expression = corrected_gene_expression
		self.normalized_corrected_ge = normalize_total_reads(corrected_gene_expression)


	def compute_ptrs(self):
		from src.peak_to_trough import compute_quantile_ptr

		ge_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
			0.2, 0.8), 1, self.normalized_reads_data.values)
		corrected_ge_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.normalized_corrected_ge.values)

		ge_comparison_ptr_df = pd.DataFrame({
			"raw_ptr": ge_ptrs,
			"corrected_ptr": corrected_ge_ptrs,
		}, index=self.normalized_reads_data.index)

		# Select only genes in our gene set (filtered for low read count)
		self.ge_comparison_ptr_df = ge_comparison_ptr_df.loc[self.repl_profile.index]
		self.ge_comparison_ptr_df['difference'] = \
			self.ge_comparison_ptr_df.corrected_ptr - self.ge_comparison_ptr_df.raw_ptr
		self.ge_comparison_ptr_df = self.ge_comparison_ptr_df.join(
			self.repl_profile[['replication_time']])

	def plot_ptrs(self, orfs=None):
		fig = plt.figure(figsize=(5, 5))
		plt.subplots_adjust(left=0.15, bottom=0.15, top=0.8)

		dat = self.ge_comparison_ptr_df

		if orfs is not None:
			dat = dat.loc[orfs]

		plt.scatter(dat.raw_ptr, dat.corrected_ptr, s=1,
			c=dat.replication_time, cmap='Spectral', vmin=5, vmax=15)
		plt.title(f"Copy # correction,\nn={len(dat)}\n{self.config_type.title()} model, Replicate {self.replicate}")
		plt.xlabel("Raw expression PTR")
		plt.ylabel("Copy-number-corrected expression PTR")

		cbar = plt.colorbar()
		cbar.ax.set_ylabel("Repl. time", rotation=270, va='bottom')
		plt.plot([0, 2], [0, 2], lw=1, ls='dotted', c='gray')

		plt.xlim(0.99, 1.2)
		plt.ylim(0.99, 1.2)
		return fig

	def plot_corrected_gene(self, gene_or_orfname, fig=None):
		from src.sgd import get_gene_name_orf_name

		orf_name, gene_name = get_gene_name_orf_name(gene_or_orfname)
		gene = self.genes.loc[orf_name]

		if fig is None:
			fig = plt.figure(figsize=(4, 3))
		plt.plot(self.normalized_reads_data.loc[gene.name], label="Original")
		plt.plot(self.corrected_gene_expression.loc[gene.name], ls='dotted', label="Corrected")
		plt.plot(self.normalized_corrected_ge.loc[gene.name], ls='dotted', label="Corrected+Normalized")

		from src.sgd import get_gene_title_name
		title = get_gene_title_name(orf_name)

		plt.title(title, pad=10)
		plt.legend()


	def save_correction(self, save_dir):

		correction_scaling = self.normalized_reads_data / self.normalized_corrected_ge

		save_path = f"{save_dir}/normalized_corrected_expression_rep{self.replicate}_{self.config_type}.csv"
		self.normalized_corrected_ge.to_csv(save_path)
		print(f"Saved to {save_path}")

		save_path = f"{save_dir}/correction_scaling_expression_rep{self.replicate}_{self.config_type}.csv"
		correction_scaling.to_csv(save_path)
		print(f"Saved to {save_path}")

		save_path = f"{save_dir}/expression_ptr_comparison_rep{self.replicate}_{self.config_type}.csv"
		self.ge_comparison_ptr_df.to_csv(save_path)
		print(f"Saved to {save_path}")


def normalize_total_reads(read_data, total_counts=60000):
	normalized_reads = read_data / \
		read_data.sum(axis=0).values.reshape((1, -1)) * total_counts
	return normalized_reads


def get_gene_replication_profile(orf_name, repl_profile=None, geneset=None):
	"""Get the replication profile for a given orf. Note the profile returned
	using the Delta-DG1 model.
	
	Loads the deconvolved profile for the given gene orf, from the gene's 
	assigned bin
	
	Returns replication index
	"""

	if repl_profile is None:
		from src.stepwise_replicatio_solver import load_replication_profile
		repl_profile = load_replication_profile()

	if geneset is None:
		from src.geneset import get_deconvolved_geneset
		geneset = get_deconvolved_geneset()

	gene = geneset.loc[orf_name]
	chrom = gene.chr
	start_indices = repl_profile.loc[chrom].index.values

	chrom_repl_profile = repl_profile.loc[chrom]
	bin_idx, bin_start_bp = get_bin_for_position(gene.TSS, start_indices)
	replication_index = chrom_repl_profile.loc[bin_start_bp].values[0]

	return replication_index


def copy_number_correct_H_index(H, data_to_correct, replication_idx):

	"""Compute the copy number correction. First compute the proportion of replicated DNA
	for this segment of the genome, by combining H (which contains the entire mixture of cells at
	each timepoint) and the replication profile (the index in H in which the gene's local genome
	has been replicated).

	Procedure:
	1. Identify the precomputed replication index in H
	2. Collect the subset of columns in H that signify replicated DNA and sum into a proportions over time
	3. The replicated proportions over time are then used to compute how much of the data to scale down to 
		1 copy of DNA.
	"""
	
	# Within the H matrix, identify the subset of
	# postG1 that will be copy number 2
	# This will be from the replication index up until the last index 
	# (last index is halted cells)
	copy_2_H = H[:, replication_idx:-1]

	# The proportion of copy number 2 cells
	# is the sum of the columns of the subset. The columns are the individual
	# deconvolved timings for f, and we are deconvolving the raw data so
	# we are only concerned about the rows
	prop_cop2 = copy_2_H.sum(axis=1)

	return copy_number_correct(prop_cop2, data_to_correct)
	

def copy_number_correct(prop_cop2, data_to_correct):
	"""Correct the data by a predefined proportion of copy number 2 at each timepoint
	index in the data_to_correct vector.

	Parameters:
		prop_cop2: 2d float array of proportions [0, 1.] of how much of population has been replicated 
		(has a copy number of 2) at each timepoint.
		data_to_correct: 1d float array of gene expression/chromatin metrics to correct
	"""

	prop_cop1 = 1 - prop_cop2
	
	# Correct by, taking the proportion of copy number 2 data
	# and dividing by 2
	data_cop2 = data_to_correct * prop_cop2 * 0.5
	data_cop1 = data_to_correct * prop_cop1

	# Recombine the copy number 1 and copy number 2 (corrected)
	data_corrected = data_cop1 + data_cop2

	return data_corrected
	

def load_expression_copy_correction(config_type, replicate):
	path = f'data/copy_correction/gene_expression/correction_scaling_expression_rep{replicate}_{config_type}.csv'
	correction = pd.read_csv(path).set_index('orf_name')
	return correction
