
# Let's analyze the first set of genes that have been deconvolved. Create a volcano plot
# get get an understanding of the mother daughter differences

from src.geneset import get_deconvolved_geneset
from glob import glob
from src.sgd import get_gene_name_orf_name
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from src.config import load_default_expression_configs


class ExpressionAnalysis(object):
	"""docstring for ExpressionAnalysis"""
	def __init__(self, output_directory):
		self.output_directory = output_directory
		self.deconvolved_genes_F = load_deconvolved_gene_expression(self.output_directory)
		self.config1, self.config2 = load_default_expression_configs()


	def plot_volcano_cg1_dg1(self, genes_callout=['DSE1', 'DSE2', 'DSE3', 'DSE4', 'SIC1', 'CLN3', 'SSK22',
			'CBF1']):

		self.cg1_dg1_data = plot_volcano_cg1_dg1(self.deconvolved_genes_F, self.config1,
										   genes_callout=genes_callout)
		plt.suptitle("Mother vs Daughter\nexpressed genes")

		thresholded_data = filter_threshold_genes(self.cg1_dg1_data, cg1_dg1_boundary_func)

		# create boundaries to separate, DG1, CG1 and no difference genes
		xs = np.linspace(0.0001, 4, 10000)
		ys = cg1_dg1_boundary_func(xs)

		plt.plot(-xs, ys, c='red', lw=0.75, ls='solid')
		plt.plot(xs, ys, c='red', lw=0.75, ls='solid')
		plt.xlim(-4, 4)

		self.dg1_dat = thresholded_data.loc[thresholded_data.meets_dg1_threshold]
		self.cg1_dat = thresholded_data.loc[thresholded_data.meets_cg1_threshold]
		self.control_dat = thresholded_data.loc[(~thresholded_data.meets_dg1_threshold) &
										   (~thresholded_data.meets_cg1_threshold)]

	def define_subsets():
		from src.sgd import get_orfnames

		# From the expression analysis
		dg1_orfs = analysis.dg1_dat.index.values
		cg1_orfs = analysis.cg1_dat.index.values
		control_orfs = analysis.control_dat.index.values
		control_orfs = list(set(control_orfs).intersection(small_prom_ratio_df.index.values))

		off_tpm = 2
		off_orfs = analysis.cg1_dg1_data.loc[analysis.cg1_dg1_data.average_TPM < off_tpm].index.values
		off_orfs = list(set(off_orfs).intersection(small_prom_ratio_df.index.values))

		high_tpm = 1000
		high_orfs = analysis.cg1_dg1_data.loc[analysis.cg1_dg1_data.average_TPM > high_tpm]\
		    .index.values
		high_orfs = list(set(high_orfs).intersection(small_prom_ratio_df.index.values))

		np.random.seed(123)
		random_orfs = np.random.choice(small_promoter_occupancies_df.index, size=200)



def load_deconvolved_gene_expression(output_directory):
	genes = get_deconvolved_geneset()

	deconvolved_expression_filenames = glob(f"{output_directory}/genes_deconvolution/*.npy")

	gene_expression_Fs_list = []
	gene_names = []
	for i, filename in enumerate(deconvolved_expression_filenames):
		expression_F = np.load(filename)
		gene_name = filename.split('/')[-1].split('_')[0]
		orf_name, gene_name = get_gene_name_orf_name(gene_name)        
		gene_expression_Fs_list.append(expression_F)
		gene_names.append(orf_name)

	expression_Fs_df = pd.DataFrame(gene_expression_Fs_list, index=gene_names)

	return expression_Fs_df


def plot_volcano_cg1_dg1(expression_Fs_df, config1, genes_callout=[]):

	from scipy.stats.distributions import norm

	# Unlog the expression data for plotting
	average_TPM = 2**np.mean(expression_Fs_df, axis=1)
	cg1_dg1_max_ratio = np.log2(expression_Fs_df[config1.cg1_indices()].mean(axis=1) / \
		expression_Fs_df[config1.dg1_indices()].mean(axis=1))

	plot_data = pd.DataFrame({'average_TPM': average_TPM, 
		'max_ratio': cg1_dg1_max_ratio+ norm.rvs(0, 0.002, len(average_TPM))})

	from src.marginal_scatter_plot import ScatterChromatinPlot
	marginal_scatter_plot = ScatterChromatinPlot()

	fig = marginal_scatter_plot.plot(
	    dat=plot_data,
	    x_key='max_ratio',
	    y_key='average_TPM',
	    highlight_genes=genes_callout,
	    xlim=(-4, 4),
	    ylim=(-10, 600),
	    orf_groups=[],
	    plot_fit=False,
	    cmap='viridis_r',
	    bw=[0.2, 0.03],
	    xlabel="$\\log_2$ [ CG1 occupancy / DG1 occupancy ]",
	    ylabel="Average deconvolved, TPM"
	)

	return plot_data


def cg1_dg1_boundary_func(xs):
	"""Boundary function to place threshold on cg1/dg1 analysis plot"""

	from scipy.stats.distributions import gamma
		
	x_offset = -0.5
	y_offset = 10
	
	ys = gamma.pdf(xs+x_offset, 1, loc=0, scale=0.2)*20+y_offset
	ys[xs < -x_offset] = 1000 # If less than the offset, set to some max value

	return ys

def filter_threshold_genes(cg1_dg1_data, cg1_dg1_boundary_func):
	xs = cg1_dg1_data.max_ratio

	neg_ratios = xs < 0
	pos_ratios = xs > 0

	neg_ys = cg1_dg1_boundary_func(-xs[neg_ratios])
	pos_ys = cg1_dg1_boundary_func(xs[pos_ratios])

	boundary_check = cg1_dg1_data.copy()
	boundary_check.loc[neg_ratios, 'boundary_y'] = neg_ys
	boundary_check.loc[pos_ratios, 'boundary_y'] = pos_ys

	boundary_check['meets_dg1_threshold'] = False
	boundary_check['meets_cg1_threshold'] = False

	boundary_check.loc[(neg_ratios) & (boundary_check.average_TPM > boundary_check.boundary_y), 
		'meets_dg1_threshold'] = True

	boundary_check.loc[(pos_ratios) & (boundary_check.average_TPM > boundary_check.boundary_y), 
		'meets_cg1_threshold'] = True

	boundary_check[boundary_check.meets_dg1_threshold]
	boundary_check[boundary_check.meets_cg1_threshold]
	return boundary_check

