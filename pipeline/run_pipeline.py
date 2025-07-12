
import sys
sys.path.append('.')

from src.figure_configs import save_figure_for_paper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import mkdirs_safe, parse_bool, print_fl

from src.combined_chromatin_model import CombinedChromatinModel
from src.chromatin_model import ChromatinModel
from src.config import load_default_chrom_configs, load_cloccs_configs

# Global parameters
WINDOWS_ALL_10K_PATH = "data/reference_data/sacCer3_genome_10k_windows.csv"
DEFAULT_CHROM_GAMMA = 0.2
DEFAULT_CHROM_KAPPA = 0.01
DEFAULT_CHROM_ETA = 0.264

# Expression parameters
DEFAULT_TX_KAPPA = 0.001
DEFAULT_TX_ETA = 0.265

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

	elif command == 'construct_rna_intermediate_files':

		from src.rna_seq_intermediates import RNASeqIntermediateManager

		# BAM files are loaded per timepoint, having reads and pileup per chromosome
		# will speed up analysis and plotting
		(_, command, output_directory, chrom) = system_args
		chrom = int(chrom)

		# Initialize for a specific chromosome
		manager = RNASeqIntermediateManager(output_directory=output_directory, 
			chromosome=chrom)

		# Save all intermediate files for the chromosome
		manager.save_chromosome_data(on_cluster=True)

		# Check if files exist
		if manager.files_exist():
			print("All files saved successfully!")

	elif command == 'call_transcripts':

		from pipeline.transcripts_caller_runner import TranscriptCallerRunner

		(_, command, output_directory) = system_args

		from src.timer import Timer

		timer = Timer()
		# Retrieve transcript calls for all of the chromosomes
		chromosomes = range(1, 17)
		all_chrom_transcripts_arr = []
		for chrom in chromosomes:
			print_fl(f"Defining transcripts for chromosome {chrom}")
			antisense_runner = TranscriptCallerRunner(
				output_directory=output_directory,
				chromosome=chrom
			)
			chrom_transcripts_results = antisense_runner.run()
			all_chrom_transcripts_arr.append(chrom_transcripts_results)
			timer.print_time()

		all_chrom_transcripts_df = pd.concat(all_chrom_transcripts_arr)

		# Following transcript calling, we have a new dataset of transcripts to compute transcription
		# for, this includes updated TSSes and non-genic transcripts for TPM calculation and deconvolution
		from src.transcripts_dataset import TranscriptDatasetBuilder

		builder = TranscriptDatasetBuilder(output_directory)

		# Load transcripts from called transcripts run for each chrom
		builder.load_transcripts(all_chrom_transcripts_df)

		# Update the sgd genes set with the updated TSS calls
		builder.update_sgd_tss_annotations()

		# Create a dataset for non-genic transcripts
		builder.define_nongenic_dataset()

		# Save each to disk
		builder.save_results()

	elif command == 'compute_tpms':

		# Compute the TPMs for the gene and non-genic sets
		(_, command, output_directory) = system_args

		from src.transcripts_dataset import load_transcripts_sets
		from src.tpm_calculator import TPMGenerator
		from src.read_bam import get_rna_seq_filepaths_df

		combined_gene_nongenic = load_transcripts_sets(output_directory, combined=True)

		tpm_generator = TPMGenerator(combined_gene_nongenic, output_directory)

		bam_df = get_rna_seq_filepaths_df(on_cluster=True)
		tpm_generator.process_multiple_replicates(bam_df)

	elif command == 'find_gamma_chromatin':

		(_, command, output_directory, index) = system_args
		index = int(index)

		from src.chromatin_gamma_search import find_gamma_chromatin

		save_directory = f"{output_directory}/chromatin_gamma_100/"
		mkdirs_safe([save_directory])

		# Select from random windows
		genome_random_100_windows_path = 'data/reference_data/sacCer3_genome_random_1k_windows.csv'

		# Set gamma to -1, meaning find the optimal gamma
		find_gamma_chromatin(save_directory, genome_random_100_windows_path, index,
			copy_correct=False, kappa=1, eta=0)

	elif command == 'find_kappa_chromatin':

		(_, command, output_directory, index) = system_args
		index = int(index)

		save_directory = f"{output_directory}/chromatin_kappa_100/"
		mkdirs_safe([save_directory])

		# Select from random windows
		genome_random_100_windows_path = 'data/reference_data/sacCer3_genome_random_1k_windows.csv'
		genome_random_100_windows = pd.read_csv(genome_random_100_windows_path)

		# Find kappa chromatin
		from src.chromatin_find_kappa import ChromatinKappaOptimizer

		# Initialize for a specific gamma value
		optimizer = ChromatinKappaOptimizer(
			output_dir=output_directory,
			save_dir=save_directory,
			gamma=0.05
		)

		row = genome_random_100_windows.loc[index]
		chrom = row.chr
		span = row.start, row.end+1

		print_fl(f"Finding optimal kappa for index:{index}, chr{chrom}, {span[0], span[1]}")

		optimizer.set_chromosome_span(chrom, span)
		results = optimizer.find_optimal_kappa_focused(
		kappa_min=1e-5, 
		kappa_max=1.0,
		verbose=True,
		save_results=True)

	elif command == 'find_eta_chromatin':

		(_, command, output_directory, index) = system_args
		index = int(index)

		save_directory = f"{output_directory}/chromatin_eta_100/"
		mkdirs_safe([save_directory])

		# Select from random windows
		genome_random_100_windows_path = 'data/reference_data/sacCer3_genome_random_1k_windows.csv'
		genome_random_100_windows = pd.read_csv(genome_random_100_windows_path)

		# Find eta chromatin
		from src.chromatin_find_eta import EtaOptimizer

		# Initialize with fixed gamma/kappa
		# Gamma and kappa are TBD
		optimizer = EtaOptimizer(output_directory, save_directory, gamma=0.05,
			kappa=0.01)

		row = genome_random_100_windows.loc[index]
		chrom = row.chr
		span = row.start, row.end+1

		print_fl(f"Finding optimal eta for index:{index}, chr{chrom}, {span[0], span[1]}")

		# Set region
		optimizer.set_chromosome_span(chrom=chrom, mnase_span=span)

		# Find optimal eta
		optimal_eta, results_df = optimizer.find_optimal_eta_linear()

	elif command == 'find_alpha':

		from src.alpha_search import FindAlphaSweepDS

		(_, command, output_directory) = system_args

		gene_names = ['DSE1', 'DSE2', 'DSE3', 'DSE4', 'ASH1', 'EGT2', 'AMN1', 
			'PRY3', 'SCW11', 'CTS1']

		finder = FindAlphaSweepDS(output_directory=output_directory, 
			gene_names=gene_names)
		finder.run_alpha_sweep(alphas=np.arange(4, 48, 2))
		finder.plot_and_save_results()

		# Compute the dg1/mg1 ratios
		finder.compute_dg1_mg1_ratios()

		# Create plots for the threshold analysis
		finder.threshold_ratios(1.6)
		finder.plot_ratio_analysis()

		# Compute the correlation between the two replicate results
		finder.compute_correlations()
		finder.plot_correlation_heatmap()

		# Sort optimal pairs
		all_pairs = finder.find_optimal_alpha_pairs()

		# Deconvolve and create figures of combined model results, expression
		finder.run_combined_gene_deconvolution(top_n_pairs=10)
		finder.save_combined_gene_results()
		fig = finder.save_combined_gene_figures()

		# Deconvolve and create figures of combined model results, origin footprint
		finder.run_combined_origin_deconvolution(top_n_pairs=10)
		finder.save_combined_origin_results()
		fig = finder.save_combined_origin_figures()

	# 2. Compute combined replication profiles for all chromosomes
	elif command == 'combined_replication':

		print_fl(f"Generating replication profiles for all chromosomes")
		(_, command, output_directory) = system_args

		from src.CombinedReplicateDeconvolutionRunner import fit_combined_replication

		single_replication_directory = None
		combined_replication_directory = f"{output_directory}/combined_replication"
		mkdirs_safe([combined_replication_directory])

		# Go straight into creating the combined replication profiles from the cloccs
		# fits, no epochs to learn parameters, nor usage of the single replication profile
		config1, config2 = load_cloccs_configs()

		num_epochs = 1
		print(f"Number of epochs {num_epochs}")
		for chrom in range(1, 17):
			print_fl(f"Chromosome {chrom}")
			combined_runner = fit_combined_replication(chrom, num_epochs,
				combined_replication_directory, config1, config2)

	elif command == 'deconvolve_expression_index':

		from src.geneset import get_deconvolved_geneset
		from pipeline.CombinedDeconvolveGeneExpressionRunner import CombinedDeconvolveGeneExpressionRunner
		from src.transcripts_dataset import load_transcripts_sets
		from src.timer import Timer
		import cvxpy as cp

		print_fl(f"Deconvolve gene expression for gene index")
		(_, command, output_directory, index) = system_args
		index = int(index)

		combined_transcripts_set = load_transcripts_sets(output_directory, combined=True)

		# Kappa and eta are empirical values. todo: methodology to identify proper parameter selection
		kappa = DEFAULT_TX_KAPPA
		eta = DEFAULT_TX_ETA

		timer = Timer()

		save_genes_directory = f"{output_directory}/genes_deconvolution/"
		mkdirs_safe([save_genes_directory])

		runner = CombinedDeconvolveGeneExpressionRunner(output_directory)

		transcript_row = combined_transcripts_set.iloc[index]
		transcript_name = transcript_row.name

		print_fl(f"[{index}/{len(combined_transcripts_set)}] Deconvolving {transcript_name}", end="...")
			
		try: 
			expression_find_gamma = runner.deconvolve_transcript_optimal_gamma(transcript_name, 
				kappa=kappa, eta=eta)
		except cp.error.SolverError:
			print_fl(f"  Failed. Skipping. {timer.get_time()}")
			return

		runner.save_to_disk(save_genes_directory)
		print_fl(f"Done. {timer.get_time()}")

	# 3. Deconvolve the gene expression for all genes
	elif command == 'deconvolve_expression':

		from src.geneset import get_deconvolved_geneset
		from pipeline.CombinedDeconvolveGeneExpressionRunner import CombinedDeconvolveGeneExpressionRunner
		from src.transcripts_dataset import load_transcripts_sets
		from src.timer import Timer
		import cvxpy as cp

		print_fl(f"Deconvolve gene expression for all genes")
		(_, command, output_directory) = system_args

		combined_transcripts_set = load_transcripts_sets(output_directory, combined=True)

		# Kappa and eta are empirical values. todo: methodology to identify proper parameter selection
		kappa = DEFAULT_TX_KAPPA
		eta = DEFAULT_TX_ETA

		timer = Timer()

		save_genes_directory = f"{output_directory}/genes_deconvolution/"
		mkdirs_safe([save_genes_directory])

		runner = CombinedDeconvolveGeneExpressionRunner(output_directory)

		index = 0
		for transcript_name, transcript_row in combined_transcripts_set.iterrows():

			index += 1

			print_fl(f"[{index}/{len(combined_transcripts_set)}] Deconvolving {transcript_name}", end="...")
			
			try: 
				expression_find_gamma = runner.deconvolve_transcript_optimal_gamma(transcript_name, 
					kappa=kappa, eta=eta)
			except cp.error.SolverError:
				print_fl(f"  Failed. Skipping.")
				continue

			runner.save_to_disk(save_genes_directory)
			print_fl(f"Done. {timer.get_time()}")

	elif command == 'deconvolve_chromatin':

		(_, command, output_directory, index) = system_args

		chromatin_save_directory = f"{output_directory}/chromatin_deconvolution/"
		mkdirs_safe([chromatin_save_directory])
		index = int(index)

		chrom, span = parse_windows_csv(WINDOWS_ALL_10K_PATH, index)
		deconvolve_chromatin(chromatin_save_directory, chrom, span,
			kappa=DEFAULT_CHROM_KAPPA, gamma=DEFAULT_CHROM_GAMMA, eta=DEFAULT_CHROM_ETA)

	elif command == 'deconvolve_chromatin_no_copy':

		(_, command, output_directory, index) = system_args
		chromatin_save_directory = f"{output_directory}/chromatin_deconvolution_no_copy/"
		mkdirs_safe([chromatin_save_directory])

		# Exact settings, with copy correction turned off
		index = int(index)
		chrom, span = parse_windows_csv(WINDOWS_ALL_10K_PATH, index)
		deconvolve_chromatin(chromatin_save_directory, chrom, span,
			copy_correct=False, kappa=DEFAULT_CHROM_KAPPA, gamma=DEFAULT_CHROM_GAMMA, 
			eta=DEFAULT_CHROM_ETA)

	elif command == 'compute_chromatin_metrics':

		(_, command, output_directory) = system_args
		metrics_save_directory = f"{output_directory}/chromatin_metrics/"
		mkdirs_safe([metrics_save_directory])

		from pipeline.chromatin_metrics_processor import ChromatinMetricsProcessor

		# Chromatin data setup, create datasets for chromatin measures
		processor = ChromatinMetricsProcessor(output_directory)
		processor.setup_data_loaders()

		# Compute chromatin metrics and save to disk
		processor.compute_chromatin_metrics_all_data(debug=False)
		processor.save_chromatin_metrics_to_disk(metrics_save_directory)

	elif command == 'parameter_summary':

		(_, command, output_directory) = system_args
		run_gamma_kappa_eta_summary(output_directory)

	# elif command == 'expression_chromatin_analysis':
		# Deprecated command, see old_pipeline.py

	elif command == 'gene_chromatin_analysis':

		pass
	
	elif command == 'promoter_analysis':

		from pipeline.expression_analysis import ExpressionAnalysis
		from src.DG1Analysis import DG1Analysis
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

	elif command == "figure1_chromatin_deconvolution":

		from src.Figure1_deconvolution import Figure1Deconvolution
		from src.figure_configs import save_figure_for_paper
		from src.utils import mkdir_safe

		(_, command, output_dir) = system_args

		fig1 = Figure1Deconvolution(output_dir=output_dir)
		fig1.run_and_save_all()
		fig1.create_panel()
		
	elif command == 'figure2_replication':

		# Rename to figure 3
		from src.Figure3_Replication import Figure3ReplicationDeconvolution
		
		(_, command, output_dir) = system_args

		fig2 = Figure2ReplicationDeconvolution(output_dir)
		fig2.setup_data()
		fig2.plot_N_G_Fr_B_components()
		fig2.plot_GNHFrB_diagram()
		fig2.plot_diagram_replication()
		fig2.layout_panel()

	elif command == 'figure3_loci':
		
		(_, command, output_dir) = system_args

		from src.Fig2LocusVignette import LocusVignette
		fig3 = LocusVignette(output_dir)
		fig3.load_regions()
		fig3.define_gene_configurations()
		fig3.compute_and_store_regions_all_analyses()
		fig3.run_and_save_all()
		fig3.layout_panel()

	elif command == 'figure4_copy_correction':

		from pipeline.figure4_copy_correction import Figure4CopyCorrection

		(_, command, output_dir) = system_args

		fig4 = Figure4CopyCorrection(output_directory=output_dir)
		fig4.load_copy_correction()
		fig4.plot_all()
		fig4.layout_panel()
		from IPython.display import Image, display

		display(Image(f'{fig4.save_dir}/Figure4_Copy_Correction_debug.png'))

	else:

		raise ValueError(f"Invalid command" + command)

	# Fit the cell cycle parameters using the previous individual replicate fits
	# as initial parameters
	# fit_combined_replication_profile(chrom=4, num_epochs=1)

	# Generate replication profiles for all chromosomes
	# generate_replication_profiles()


def deconvolve_chromatin(chromatin_save_directory, chrom, span,
	copy_correct=True, gamma=0.0066, kappa=1, eta=0):

	# Load the configs from disk
	config1, config2 = load_default_chrom_configs()

	enable_find_gamma = gamma < 0

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
		save_figure_for_paper(f"{raw_plots_directory}/raw_rep1_{save_title}.png")
		plt.close(fig)

		fig = combined_model.chrom2_model.plot_raw_data(figsize=(11, 11))
		save_figure_for_paper(f"{raw_plots_directory}/raw_rep2_{save_title}.png")
		plt.close(fig)

		combined_model.setup_deconv_model(copy_correct=copy_correct)

		# Find optimum run, just save gamma and elbow
		if enable_find_gamma:
			combined_model.deconvolve_find_optimal_gamma(kappa=kappa, eta=eta,
				verbose=True)	

			# Save the gamma and elbow curve to disk
			save_gamma_path = f"{data_directory}/{save_title}_gamma.txt"
			with open(save_gamma_path, 'w') as save_file:
				save_file.write(f"{combined_model.gamma}")

			save_elbow_path = f"{deconv_plots_directory}/{save_title}_elbow.png"
			combined_model.find_gamma_chromatin.plot_gamma_sweep()
			save_figure_for_paper(save_elbow_path)

		# Normal run, deconvolve and save results
		else:
			combined_model.deconvolve(gamma=gamma, kappa=kappa, eta=eta,
				verbose=True)	
			np.save(f"{data_directory}/{save_title}_F.npy", combined_model.F)
			combined_model.plot_branches(figsize=(50, 11))
			save_figure_for_paper(f"{deconv_plots_directory}/deconv_{save_title}.png")
			plt.close(fig)

	deconv_and_save(chrom, span, chromatin_save_directory)

def parse_windows_csv(path, index):

	# Read the window datas data and load the relevant row
	window_set = pd.read_csv(path)
	row = window_set.iloc[index]

	chrom = row.chr
	span = row.start, row.end+1 # (Add 1 to include the last base)
	print_fl(f"Deconvolving index:{index}, chr{chrom}, {span[0], span[1]}")

	return chrom, span

def run_gamma_kappa_eta_summary(output_directory):
	from glob import glob
	gammas = []
	for filepath in glob(f'{output_directory}/chromatin_gamma_100/deconvolution_data/chr*/*.txt'):
		with open(filepath, 'r') as f:
			gamma = float(f.read())
			gammas.append(gamma)

	from glob import glob
	kappas = []
	for filepath in glob(f'{output_directory}/chromatin_kappa_100/*.csv'):
		kappa = pd.read_csv(filepath).iloc[0].kappa
		kappas.append(kappa)

	from glob import glob
	etas = []
	for filepath in glob(f'{output_directory}/chromatin_eta_100/*.txt'):
		eta = pd.read_csv(filepath)
		
		with open(filepath, 'r') as f:
			for line in f.readlines():
				if line.startswith('  Optimal eta'):
					eta = float(line.split(' ')[-1])
		etas.append(eta)

	import statistics
	from src.helpers import find_mode_float

	if len(gammas) > 0:
		optimal_gamma = statistics.mode(gammas)
	else:
		optimal_gamma = None

	if len(etas) > 0:
		optimal_eta = statistics.mode(etas)
	else:
		optimal_eta = None

	plt.figure(figsize=(9, 2.5))
	plt.subplot(1, 3, 1)
	plt.hist(gammas, bins=40)
	plt.axvline(optimal_gamma, c='red', lw=1.5, ls='dotted')
	# plt.xlim(0, 0.1)
	plt.title(f"$\\gamma^*=${optimal_gamma:.3f}")
	plt.xlabel("$\\gamma$")
	plt.ylabel("Frequency")

	plt.subplot(1, 3, 2)


	# if len(kappas) > 0:
	# 	kappas = np.array(kappas)
	# 	optimal_kappa = statistics.mode(kappas)
	# 	# Let the function create its own bins
	# 	kappa_bin_width = 0.001
	# 	optimal_kappa = find_mode_float(kappas, kappa_bin_width)
	# else:
	# 	optimal_kappa = None

	# # Use the same number of bins for matplotlib
	# n_bins = int((kappas.max() - kappas.min()) / kappa_bin_width)
	# plt.hist(kappas, bins=n_bins)
	# plt.axvline(optimal_kappa, c='red', lw=1.5, ls='dotted')
	# plt.title(f"$\\kappa^*=${optimal_kappa:.3f}")
	# plt.xlabel("$\\kappa$")

	# plt.subplot(1, 3, 3)
	# plt.hist(etas, bins=20)
	# plt.xlim(0, 4)
	# plt.axvline(optimal_eta, c='red', lw=1.5, ls='dotted')
	# plt.title(f"$\\eta^*=${optimal_eta:.3f}")
	# plt.xlabel("$\\eta$")

	plt.suptitle("Optimal regularization parameters for 100 random windows (of width 1000 bp)",
		fontweight='demi', fontsize=16)

	plt.tight_layout()

	# print(f"Optimal:\n"
	# 	  f"\tgamma:\t{optimal_gamma:.3f}\n" 
	# 	  f"\tkappa:\t{optimal_kappa:.3f}\n"
	# 	  f"\teta:\t{optimal_eta:.3f}")

	# return kappas




if __name__ == '__main__':
	main()
