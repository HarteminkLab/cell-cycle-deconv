
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.chromatin_model import ChromatinModel
from src.CombinedReplicationDeconvolution import load_config_from_replication_runs
from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis

from src.plot_helpers import color_for_key
dg1_color = color_for_key('DG1')
cg1_color = color_for_key('CG1')
random_color = plt.cm.gray(0.25)


class DG1Analysis:
	"""Analysis of DG1-CG1 specific expression and chromatin"""
	def __init__(self, output_directory):

		self.output_directory = output_directory
		self.config1, self.config2 = load_config_from_replication_runs(f'{output_directory}/single_replication/', 
			chrom=4)
		
		chromatin_data_dir = f"{output_directory}/chromatin_deconvolution/deconvolution_data"
		self.genome_analysis = GenomeDeconvolutionAnalysis(chromatin_data_dir)

	def compute_chromatin_measures(self):
		# todo: in another notebook, and take some time to compute
		pass

	def load_chromatin_measures(self):

		chromatin_analysis_directory = \
			f'{self.output_directory}/chromatin_analysis/'

		self.small_promoter_occupancies_df = pd.read_csv(f"{chromatin_analysis_directory}/small_prom_occupancies.csv").set_index('orf_name')
		self.nucleosome_genebody_entropies_df = pd.read_csv(f"{chromatin_analysis_directory}/nuc_gb_entropies.csv").set_index('orf_name')
		self.nucleosome_gene_body_occupancies_df = pd.read_csv(f"{chromatin_analysis_directory}/nuc_gb_occupancies.csv").set_index('orf_name')
		self.small_gene_body_occupancies_df = pd.read_csv(f"{chromatin_analysis_directory}/small_gb_occupancies.csv").set_index('orf_name')

		for df in [self.small_promoter_occupancies_df, self.nucleosome_genebody_entropies_df,
				   self.nucleosome_gene_body_occupancies_df, self.small_gene_body_occupancies_df]:
			df.columns = df.columns.astype(int)

	def define_subsets(self, expression_analysis):
		"""Define the set of ORFs to perform analysis against"""

		from src.sgd import get_orfnames

		self.highlight_genes = ['DSE1', 'DSE2', 'DSE3', 'DSE4', 'SIC1', 'CLN3', 'SSK22',
						  'CBF1']
		self.highlight_orfs = get_orfnames(self.highlight_genes)

		# Define DG1 orfs from the expression set
		self.dg1_orfs = expression_analysis.dg1_dat.index.values
		self.cg1_orfs = expression_analysis.cg1_dat.index.values

		# Define random set of ORFs
		np.random.seed(123)
		self.random_orfs = np.random.choice(self.small_promoter_occupancies_df.index, size=200)

	def plot_dg1_vs_cg1(self):

		from src.marginal_scatter_plot import ScatterChromatinPlot
		from src.chromatin_metrics import generate_cg1_dg1_data_for_plotting

		small_prom_ratio_df = generate_cg1_dg1_data_for_plotting(self.config1, 
			self.small_promoter_occupancies_df)
		small_prom_diff_df = generate_cg1_dg1_data_for_plotting(self.config1, 
			self.small_promoter_occupancies_df, mode='difference')

		chromatin_plot = ScatterChromatinPlot()
		fig = chromatin_plot.plot(
			dat=small_prom_ratio_df,
			highlight_genes=self.highlight_genes,
			xlim=(-1, 1),
			ylim=(1, 4.25),
			orf_groups=[],
			xlabel="$\\log_2$ [ CG1 occupancy / DG1 occupancy ]",
			ylabel="Mean occupancy",
			cmap=plt.cm.plasma_r
		)
		plt.suptitle("Mother vs Daughter small fragments\npromoter occupancy", fontsize=12)


	def plot_subset_dg1_tx(self):
		"""Plot the DG1 selected ORFs"""
		self.plot_subset_promoter_scatter(self.dg1_orfs, dg1_color, "Daughter expressed")


	def plot_subset_promoter_scatter(self, orfs, color, title):

		from src.marginal_scatter_plot import ScatterChromatinPlot
		from src.chromatin_metrics import generate_cg1_dg1_data_for_plotting

		small_prom_ratio_df = generate_cg1_dg1_data_for_plotting(self.config1, 
			self.small_promoter_occupancies_df)

		chromatin_plot = ScatterChromatinPlot()
		ks_statistic, ks_pvalue = compute_ks_for_df(small_prom_ratio_df, self.dg1_orfs)
		fig = chromatin_plot.plot(
			dat=small_prom_ratio_df,
			highlight_genes=[],
			xlim=(-1, 1),
			ylim=(1, 4.25),
			orf_groups=[(f"DG1 expressed genes, n={len(self.dg1_orfs)}", color, self.dg1_orfs)],
			xlabel="$\\log_2$ [ CG1 occupancy / DG1 occupancy ]",
			ylabel="Mean occupancy"
		)
		plt.suptitle(f"{title}:\n"
					 f" KS={ks_statistic:.2f}, p-value={ks_pvalue:.2g}", fontsize=12)


from scipy.stats import ks_2samp

def compute_ks_for_df(small_prom_df, orfs):
	null_distribution = small_prom_df.x.values
	subset_distribution = small_prom_df.x.loc[orfs].values
	statistic, pvalue = ks_2samp(null_distribution, subset_distribution)
	return statistic, pvalue

