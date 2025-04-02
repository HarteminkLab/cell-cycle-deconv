
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

	def plot_subset_cg1_tx(self):
		"""Plot the DG1 selected ORFs"""
		self.plot_subset_promoter_scatter(self.cg1_orfs, cg1_color, "Mother expressed")

	def plot_subset_promoter_scatter(self, orfs, color, title):

		from src.marginal_scatter_plot import ScatterChromatinPlot
		from src.chromatin_metrics import generate_cg1_dg1_data_for_plotting

		small_prom_ratio_df = generate_cg1_dg1_data_for_plotting(self.config1, 
			self.small_promoter_occupancies_df)

		chromatin_plot = ScatterChromatinPlot()
		ks_statistic, ks_pvalue = compute_ks_for_df(small_prom_ratio_df, orfs)
		fig = chromatin_plot.plot(
			dat=small_prom_ratio_df,
			highlight_genes=[],
			xlim=(-1, 1),
			ylim=(1, 4.25),
			orf_groups=[(f"DG1 expressed genes, n={len(orfs)}", color, orfs)],
			xlabel="$\\log_2$ [ CG1 occupancy / DG1 occupancy ]",
			ylabel="Mean occupancy"
		)
		plt.suptitle(f"{title}:\n"
					 f" KS={ks_statistic:.2f}, p-value={ks_pvalue:.2g}", fontsize=12)


	def compute_g1_expression_sm_correlations(self, expressions_F, config1):

		from src.helpers import compute_row_correlations

		t_indices = config1.t_indices()
		b_indices = config1.b_indices()

		# Compute the mother and daughter correlations for all genes.
		small_occupancies = self.small_promoter_occupancies_df
		t_correlations = compute_row_correlations(expressions_F[t_indices], 
		   small_occupancies.loc[expressions_F.index][t_indices])
		b_correlations = compute_row_correlations(expressions_F[b_indices], 
		   small_occupancies.loc[expressions_F.index][b_indices])

		self.small_tx_top_correlations = t_correlations
		self.small_tx_bottom_correlations = b_correlations


	def plot_tx_sm_correlation_histograms(self, mg1_only, mg1_and_dg1, dg1_only,
		postg1_only, all_other_orfs):

		t_correlations = self.small_tx_top_correlations
		b_correlations = self.small_tx_bottom_correlations

		indices_sets = ['t', 'b']
		orf_sets = [mg1_only, mg1_and_dg1, dg1_only, postg1_only, all_other_orfs]
		column_names = [f"Mother G1 expressed,\nn={len(mg1_only)}",
						f"Mother and\nDaughter G1 expressed,\nn={len(mg1_and_dg1)}", 
					   f"Daughter G1 expressed,\nn={len(dg1_only)}",
					   f"S/G2/M expressed,\nn={len(postg1_only)}",
					   f"All others,\nn={len(all_other_orfs)}"]
		row_names = ["Mother branch", "Daughter branch"]

		plt.figure(figsize=(31, 4))

		i = 1
		columns = len(column_names)
		rows = 2
		for row, index_set_name in enumerate(indices_sets):
			for col, orf_set in enumerate(orf_sets):
				plt.subplot(rows, columns, i)
				
				if index_set_name == 't':
					correlations = t_correlations.loc[orf_set]
				else:
					correlations = b_correlations.loc[orf_set]
					
				plt.hist(correlations.correlation, bins=np.linspace(-1, 1, 10))
				plt.axvline(0, c='black', lw=1, ls='dotted')
				i += 1

				if col == 4:
					plt.ylim(0, 1000)
				else:
					plt.ylim(0, 80)

				plt.xlim(-1, 1)

				if row == 0:
					plt.title(column_names[col])
					plt.xticks([])
				else:
					if col == 0: plt.xlabel("Pearson $r$")

				if col == 0:
					if row == 0:
						plt.ylabel("# genes")
				else:
					#plt.yticks([])
					pass
					
				if col == 4:
					right_ax = plt.twinx()
					right_ax.set_yticks([])
					right_ax.set_ylabel(row_names[row], ha='left', rotation=0)

		plt.subplots_adjust(left=0.3, top=0.73, right=0.7, wspace=0.35)
		plt.suptitle("Correlation of expression and promoter occupancy", fontsize=16)


from scipy.stats import ks_2samp

def compute_ks_for_df(small_prom_df, orfs):
	null_distribution = small_prom_df.x.values
	subset_distribution = small_prom_df.x.loc[orfs].values
	statistic, pvalue = ks_2samp(null_distribution, subset_distribution)
	return statistic, pvalue

