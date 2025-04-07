
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import mkdirs_safe, parse_bool, print_fl

from src.combined_chromatin_model import CombinedChromatinModel
from src.chromatin_model import ChromatinModel
from src.config import load_default_chrom_configs

def main():

	system_args = tuple(sys.argv)
	command = system_args[1]

	print(f"Pipeline arguments: " , system_args)

	# 0. Create target length distribution
	if command == 'length_distribution':

		(_, command, output_directory) = system_args

		length_replication_directory = f"{output_directory}/length_distribution"
		mkdirs_safe([output_directory, length_replication_directory])

		from src.length_distribution import LengthDistributionCalculator, \
			plot_fragment_length_distributions

		# Compute the length distributions for replicate 1
		length_dist_calculator1 = LengthDistributionCalculator(replicate=1)
		length_dist_calculator1.compute_length_distributions_for_all_chroms()
		rep1_length_distribution = length_dist_calculator1.create_rep_distribution()

		# Compute the length distributions for replicate 2
		length_dist_calculator2 = LengthDistributionCalculator(replicate=2)
		length_dist_calculator2.compute_length_distributions_for_all_chroms()
		rep2_length_distribution = length_dist_calculator2.create_rep_distribution()

		# Create a target distribution to normalize each window to
		combined_target_distribution = (rep1_length_distribution + rep2_length_distribution)/2.
		target_distribution_df = pd.DataFrame({'combined': combined_target_distribution,
			 'rep1': rep1_length_distribution,
			 'rep2': rep2_length_distribution})
		target_distribution_df.index.name = 'fragment_length'
		target_distribution_df.to_csv(f'{length_replication_directory}/target_length_distribution.csv')

		# Plot replicate 1 and 2 distributions for each sample to show similarity

		# Save the plot to disk
		plt.figure(figsize=(4, 3))
		plt.plot(target_distribution_df['rep1'], alpha=0.25, lw=3, label="Replicate 1")
		plt.plot(target_distribution_df['rep2'], alpha=0.25, lw=3, label="Replicate 2")
		plt.plot(target_distribution_df['combined'], c='black', lw=1, label="Target/Combined")
		plt.title("Target length distribution")
		plt.legend()
		plt.savefig(f"{length_replication_directory}/target_distribution.png")

		# Plot the raw data replication distribution
		plot_fragment_length_distributions(length_dist_calculator1.all_length_dists, 
			length_dist_calculator2.all_length_dists)
		plt.savefig(f"{length_replication_directory}/raw_distributions.png")


	# 1. Deconvolve individual replication profiles, learn cell cycle parameters from MNase-seq
	elif command == 'replication':

		(_, command, output_directory, replicate, chrom, num_epochs, cold_start) = system_args

		single_replication_directory = f"{output_directory}/single_replication"
		mkdirs_safe([output_directory, single_replication_directory])

		cold_start = parse_bool(cold_start)
		chrom = int(chrom)
		replicate = int(replicate)
		num_epochs = int(num_epochs)

		# 1. Compute replication profiles for each replicate using chromosome 4
		from pipeline.fit_replication_profiles import main as fit_replication_profile
		fit_replication_profile(chrom=chrom, replicate=replicate, num_epochs=num_epochs, 
			output_directory=single_replication_directory, cold_start=cold_start)

	elif command == 'replication_second_stage':

		(_, command, output_directory, replicate, chrom, num_epochs, cold_start) = system_args

		single_replication_directory = f"{output_directory}/single_replication"

		cold_start = parse_bool(cold_start)
		chrom = int(chrom)
		replicate = int(replicate)
		num_epochs = int(num_epochs)

		# 1. Compute replication profiles for each replicate using chromosome 4
		from pipeline.fit_replication_profiles import main as fit_replication_profile
		fit_replication_profile(chrom=chrom, replicate=replicate, num_epochs=num_epochs, 
			output_directory=single_replication_directory, cold_start=cold_start)


	# 2. Compute combined replication profiles for all chromosomes
	elif command == 'combined_replication':

		print_fl(f"Generating replication profiles for all chromosomes")
		(_, command, output_directory) = system_args

		single_replication_directory = f"{output_directory}/single_replication"
		combined_replication_directory = f"{output_directory}/combined_replication"
		mkdirs_safe([combined_replication_directory])

		from src.CombinedReplicateDeconvolutionRunner import main as fit_combined_replication
		from src.CombinedReplicationDeconvolution import load_config_from_replication_runs

		# In this case, we are loading the configs as a continuation from the single
		# replication convergence.
		config1, config2 = load_config_from_replication_runs(single_replication_directory, chrom=4)

		# For each chromosome, create the replication profiles for each of the chromosomes and save to disk
		# todo: currently not aiming to learn the cell cycle parameters on this step for efficiency purposes.
		num_epochs = 1
		warm_start_chrom = 4
		for chrom in range(1, 17):
			print_fl(f"Chromosome {chrom}")
			combined_runner = fit_combined_replication(chrom, num_epochs, single_replication_directory,
				combined_replication_directory, warm_start_chrom, config1, config2)

	# 3. Deconvolve the gene expression for all genes
	elif command == 'deconvolve_expression':

		from src.geneset import get_deconvolved_geneset
		from pipeline.CombinedDeconvolveGeneExpressionRunner import CombinedDeconvolveGeneExpressionRunner

		genes = get_deconvolved_geneset()

		print_fl(f"Deconvolve gene expression for all genes")
		(_, command, output_directory) = system_args

		from src.timer import Timer
		import cvxpy as cp

		timer = Timer()

		save_genes_directory = f"{output_directory}/genes_deconvolution/"
		mkdirs_safe([save_genes_directory])

		runner = CombinedDeconvolveGeneExpressionRunner(output_directory)

		index = 0
		for _, gene in genes.iterrows():

			index += 1

			gene_name = gene['gene']

			print_fl(f"[{index}/{len(genes)}] Deconvolving {gene_name}", end="...")
			
			try: 
				expression_find_gamma = runner.deconvolve_gene(gene_name)
			except cp.error.SolverError:
				print_fl(f"  Failed. Skipping.")
				continue

			runner.save_to_disk(save_genes_directory)
			print_fl(f"Done. {timer.get_time()}")

	elif command == 'deconvolve_chromatin_staging':

		(_, command, output_directory, index) = system_args
		chromatin_save_directory = f"{output_directory}/chromatin_deconvolution_staging/"
		index = int(index)

		window_set_path = "datasets/computed_mnase/test_window_set_2kb.csv"
		deconvolve_chromatin(chromatin_save_directory, window_set_path, index)

	elif command == 'deconvolve_chromatin_full':

		(_, command, output_directory, index) = system_args
		chromatin_save_directory = f"{output_directory}/chromatin_deconvolution/"
		mkdirs_safe([chromatin_save_directory])
		index = int(index)

		window_set_path = "data/reference_data/sacCer3_genome_10k_windows.csv"
		deconvolve_chromatin(chromatin_save_directory, window_set_path, index)

	elif command == 'deconvolve_chromatin_full_no_copy':

		(_, command, output_directory, index) = system_args
		chromatin_save_directory = f"{output_directory}/chromatin_deconvolution_no_copy/"
		mkdirs_safe([chromatin_save_directory])

		index = int(index)

		window_set_path = "data/reference_data/sacCer3_genome_10k_windows.csv"
		deconvolve_chromatin(chromatin_save_directory, window_set_path, index,
			copy_correct=False)

	elif command == 'promoter_analysis':

		from pipeline.expression_analysis import ExpressionAnalysis
		from src.DG1Analysis import DG1Analysis
		from src.figure_configs import save_figure_for_paper
		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis

		(_, command, output_directory) = system_args

		save_directory = f"{output_directory}/analysis/expression_promoter_analysis"
		mkdirs_safe([save_directory])

		# Load chromatin analysis
		chromatin_analysis = DG1Analysis(output_directory)
		chromatin_analysis.load_chromatin_measures()

		# Load expression analysis
		expression_analysis = ExpressionAnalysis(output_directory)

		# Plot the CG1 DG1 analysis plot
		fig = expression_analysis.plot_volcano_cg1_dg1(genes_callout=['DSE1', 'DSE2', 'DSE3', 'DSE4', 'PHO5',
		                                            'HO', 'SPL2', 'PIR1', 'EGT2', 'TOS6'])
		save_figure_for_paper(f"{save_directory}/cg1_dg1_expression.png")
		plt.close(fig)

		# Load shift analysis
		from src.expression_promoter_shift_analysis import PromoterExpressionShiftAnalysis

		# analysis and dg1 analysis are the objects that hold the tx and chromatin data
		# they should be renamed
		config1, config2 = load_default_chrom_configs()
		promoter_expression_shift_analysis = PromoterExpressionShiftAnalysis(
			expression_analysis, chromatin_analysis, config1)

		# Subset the expression groups
		fig, axs = promoter_expression_shift_analysis.plot_expression_quantiles()
		save_figure_for_paper(f"{save_directory}/expression_groups.png")
		plt.close(fig)

		# Filter low std genes
		promoter_expression_shift_analysis.compute_high_std_promoter_orfs_set()
		save_figure_for_paper(f"{save_directory}/high_std_orfs_set.png")
		promoter_expression_shift_analysis.filter_gene_subsets_for_high_std()

		# Compute correlations
		promoter_expression_shift_analysis.compute_promoter_expression_correlations()

		# Plot example gene
		fig = promoter_expression_shift_analysis.correlation_calculator.plot_gene_correlation('CLN1', 
			branch_name="daughter")
		save_figure_for_paper(f"{save_directory}/correlation_CLN1.png")

		# Plot regulation classification
		fig = promoter_expression_shift_analysis.correlation_calculator.plot_regulation_type_bar_counts("mother", 
			promoter_expression_shift_analysis.t_filtered_gene_subsets)
		save_figure_for_paper(f"{save_directory}/mother_regulators.png")
		plt.close(fig)
		fig = promoter_expression_shift_analysis.correlation_calculator.plot_regulation_type_bar_counts("daughter", 
			promoter_expression_shift_analysis.b_filtered_gene_subsets)
		save_figure_for_paper(f"{save_directory}/daughter_regulators.png")
		plt.close(fig)

		# Plot example gene context
		genome_analysis = GenomeDeconvolutionAnalysis(outdir=
		    f"{output_directory}/chromatin_deconvolution/deconvolution_data")
		fig = genome_analysis.plot_gene('DSE3', config1, expression_analysis)
		save_figure_for_paper(f"{save_directory}/locus_DSE3.png")
		plt.close(fig)

	else:


		raise ValueError(f"Invalid command" + command)

	# Fit the cell cycle parameters using the previous individual replicate fits
	# as initial parameters
	# fit_combined_replication_profile(chrom=4, num_epochs=1)

	# Generate replication profiles for all chromosomes
	# generate_replication_profiles()


def deconvolve_chromatin(chromatin_save_directory, window_set_path, index,
	copy_correct=True):

	# Deconvolve the initial set of chromatin windows for testing,
	# priority over deconvolving the most important windows first

	# Load the configs from disk
	config1, config2 = load_default_chrom_configs()

	def deconv_and_save(chrom, mnase_span, chromatin_save_directory):

		data_directory = f"{chromatin_save_directory}/deconvolution_data/chr{chrom}"
		raw_plots_directory = f"{chromatin_save_directory}/raw_plots_directory/chr{chrom}"
		deconv_plots_directory = f"{chromatin_save_directory}/deconv_plots_directory/chr{chrom}"

		mkdirs_safe([data_directory, raw_plots_directory, deconv_plots_directory])
		combined_model = CombinedChromatinModel(config1=config1, config2=config2)

		# Load window to deconvolve
		combined_model.load_mnase_span(chrom, mnase_span)

		save_title = f"chr{chrom}_{mnase_span[0]}_{mnase_span[1]}"

		# Plot normalization check
		fig = combined_model.plot_normalization_sanity_check()
		plt.savefig(f"{raw_plots_directory}/normalization_{save_title}.png")
		plt.close(fig)

		# Plot raw data
		fig = combined_model.chrom1_model.plot_raw_data(figsize=(11, 11))
		plt.savefig(f"{raw_plots_directory}/raw_rep1_{save_title}.png")
		plt.close(fig)

		fig = combined_model.chrom2_model.plot_raw_data(figsize=(11, 11))
		plt.savefig(f"{raw_plots_directory}/raw_rep2_{save_title}.png")
		plt.close(fig)
	
		combined_model.setup_deconv_model(copy_correct=copy_correct)
		combined_model.deconvolve(gamma=0.01, kappa=0, verbose=True)	

		np.save(f"{data_directory}/{save_title}_F.npy", combined_model.F)

		combined_model.plot_branches(figsize=(50, 11))
		plt.savefig(f"{deconv_plots_directory}/deconv_{save_title}.png")
		plt.close(fig)

	window_set = pd.read_csv(window_set_path)
	row = window_set.iloc[index]

	chrom = row.chr
	span = row.start, row.end+1 # (Add 1 to include the last base)

	print_fl(f"Deconvolving index:{index}, chr{chrom}, {span[0], span[1]}")

	deconv_and_save(chrom, span, chromatin_save_directory)



if __name__ == '__main__':
	main()
