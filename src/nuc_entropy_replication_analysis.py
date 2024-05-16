
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_timecourse_by_repl(metric_df, gene_chrom_repl_analysis, ylims, title, tx_cutoffs=None):

	from src.gene_chromatin_replication_analysis import plot_box_timecourse
	from src.boxplot import get_boxplot_data_for_metric_df
	from src.helpers import get_quantile_values

	geneset_entropy_exp_repl_df = gene_chrom_repl_analysis.get_entropy_vs_tx_df()

	if tx_cutoffs is not None:
		geneset_entropy_exp_repl_df = geneset_entropy_exp_repl_df[(geneset_entropy_exp_repl_df.expression >= tx_cutoffs[0]) & 
			(geneset_entropy_exp_repl_df.expression < tx_cutoffs[1])]

	segments, qvals, lens = get_quantile_values(geneset_entropy_exp_repl_df.replication_timing, 
	    q=[0.25, 0.5, 0.75])
	early_genes, early_mid_genes, mid_late_genes, late_genes = segments

	early_narrow_df = get_boxplot_data_for_metric_df(metric_df, 
		early_genes.index, 'entropy')
	earlymid_narrow_df = get_boxplot_data_for_metric_df(metric_df, 
		early_mid_genes.index, 'entropy')
	midlate_narrow_df = get_boxplot_data_for_metric_df(metric_df, 
		mid_late_genes.index, 'entropy')
	late_narrow_df = get_boxplot_data_for_metric_df(metric_df, 
		late_genes.index, 'entropy')

	plt.figure(figsize=(13, 8))
	ax = plt.subplot(4, 1, 1)
	plot_box_timecourse(ax, early_narrow_df, 0/4., "Early")
	plt.ylim(*ylims)
	plt.xlabel('')
	plt.xticks([])

	ax = plt.subplot(4, 1, 2)
	plot_box_timecourse(ax, earlymid_narrow_df, 1/4., "Early-Mid")
	plt.ylim(*ylims)
	plt.xlabel('')
	plt.xticks([])

	ax = plt.subplot(4, 1, 3)
	plot_box_timecourse(ax, midlate_narrow_df, 2/4., "Mid-Late")
	plt.ylim(*ylims)
	plt.xlabel('')
	plt.xticks([])

	ax = plt.subplot(4, 1, 4)
	plot_box_timecourse(ax, late_narrow_df, 3/4., "Late")
	plt.ylim(*ylims)
	plt.xticks([])
	plt.xlabel("")

	plt.suptitle(title, fontsize=16)
	plt.subplots_adjust(top=0.9)
