
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

		thresh_x_min, thresh_y_min = 0.25, 20

		thresholded_data = filter_threshold_genes(self.cg1_dg1_data, cg1_dg1_boundary_func,
			min_y=thresh_y_min, min_x=thresh_x_min)

		# create boundaries to separate, DG1, CG1 and no difference genes
		xs = np.linspace(0.0001, 4, 10000)
		ys = cg1_dg1_boundary_func(xs, thresh_y_min, thresh_x_min)

		plt.plot(-xs, ys, c='red', lw=0.75, ls='solid')
		plt.plot(xs, ys, c='red', lw=0.75, ls='solid')
		plt.xlim(-4, 4)

		self.dg1_dat = thresholded_data.loc[thresholded_data.meets_dg1_threshold]
		self.cg1_dat = thresholded_data.loc[thresholded_data.meets_cg1_threshold]
		self.control_dat = thresholded_data.loc[(~thresholded_data.meets_dg1_threshold) &
										   (~thresholded_data.meets_cg1_threshold)]

	def retrieve_mg1_dg1_specific_genes(self, config1, proportion_threshold):
		"""Retrieve G1 specific expression genes using a threshold. Separates into
		DG1 only, MG1 only and DG1 and MG1 expressed genes."""

		expressions_F = self.deconvolved_genes_F

		# Compute the number of MG1, DG1 specific genes and G1 specific genes.
		mg1_i = config1.cg1_indices()
		dg1_i = config1.dg1_indices()
		postg1_i = config1.postg1_indices()

		mg1_expression = expressions_F[mg1_i]
		dg1_expression = expressions_F[dg1_i]

		mean_mg1_expression = mg1_expression.mean(1)
		mean_dg1_expression = dg1_expression.mean(1)
		mean_postg1_expression = expressions_F[postg1_i].mean(1)

		orfs = expressions_F.index

		# How many genes have g1 expression mg1 union dg1
		mg1_select = mean_mg1_expression > mean_postg1_expression*(1+proportion_threshold)
		dg1_select = mean_dg1_expression > mean_postg1_expression*(1+proportion_threshold)

		# M and D have similar expression levels, use the l2 norm and set to threshold
		# these to a low value, can look at the histogram to decide on the threshold
		cg1_dg1_l2 = (((mg1_expression.values-dg1_expression.values)**2).sum(1)**0.5)
		mg1_and_dg1_similar_expression = cg1_dg1_l2 < 1

		# MG1 and DG1 are similarly expressed and G1-specific
		mg1_and_dg1_select = mg1_select & dg1_select & mg1_and_dg1_similar_expression
		mg1_and_dg1 = orfs[mg1_and_dg1_select]

		# MG1 is transcribed
		mg1_only = orfs[mg1_select & ~mg1_and_dg1_select]

		# DG1 is transcribed
		dg1_only = orfs[dg1_select & ~mg1_and_dg1_select]
		
		print(f"Threshold of {1+proportion_threshold} > S/G2/M expression. There are:")
		
		print(f"{len(mg1_only)} Mother G1 specific genes")
		print(f"{len(dg1_only)} Daughter G1 specific genes")
		print(f"{len(mg1_and_dg1)} Mother and Daughter expressed genes")

		return mg1_only, mg1_and_dg1, dg1_only, 


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

	from src.plot_helpers import create_sub_colormap
	cmap = create_sub_colormap('Purples', 0.25, 1., 'Purples_darker')

	fig = marginal_scatter_plot.plot(
		dat=plot_data,
		x_key='max_ratio',
		y_key='average_TPM',
		highlight_genes=genes_callout,
		xlim=(-4, 4),
		ylim=(-10, 600),
		orf_groups=[],
		plot_fit=False,
		cmap=cmap,
		bw=[0.2, 0.03],
		xlabel="$\\log_2$ [ CG1 occupancy / DG1 occupancy ]",
		ylabel="Average deconvolved, TPM"
	)

	return plot_data


def cg1_dg1_boundary_func(xs, y_offset=10, x_offset=0.5,
	scale=0.2, multiplier=100):
	"""Boundary function to place threshold on cg1/dg1 analysis plot"""

	from scipy.stats.distributions import gamma
	
	ys = gamma.pdf(xs-x_offset, 1, loc=0, scale=scale)*multiplier+y_offset
	ys[xs < +x_offset] = 1e9 # If less than the offset, set to some max value

	return ys


def filter_threshold_genes(cg1_dg1_data, cg1_dg1_boundary_func, min_y=10, min_x=0.5):
	xs = cg1_dg1_data.max_ratio

	neg_ratios = xs < 0
	pos_ratios = xs > 0

	neg_ys = cg1_dg1_boundary_func(-xs[neg_ratios], min_y, min_x)
	pos_ys = cg1_dg1_boundary_func(xs[pos_ratios], min_y, min_x)

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

