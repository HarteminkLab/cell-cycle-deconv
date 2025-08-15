import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.transcripts_dataset import load_transcripts_sets

# from typing import List, Tuple, Optional, Union
# import warnings
from src.plot_helpers import plot_rect2


def plot_tx_calls(ax, nongenic, chrom, span, color_strand=False):
	
	# Expand search span to find whole transcripts (for visual plotting)
	search_span = span[0]-2000, span[1]+2000
	selected = nongenic[(nongenic.chr == chrom) &
		(nongenic.start > search_span[0]) & (nongenic.stop < search_span[1])]

	for _, row in selected.iterrows():
		y0 = 0
		y1 = 30
		if row.strand == '-':
			y1 = -30
			
		if color_strand:
			color = plt.cm.Reds(0.1) if row.strand == '-' else plt.cm.Blues(0.1)
		else:
			color = '#ddd'
		plot_rect2(ax, row.TSS, y0, row.PAS, y1, color=color, zorder=0)
		
def plot_genic_non_genic_transcript_calls(ax, chrom, span, outdir):
	genic, nongenic = load_transcripts_sets(outdir)
	plot_tx_calls(ax, nongenic, chrom, span)
	plot_tx_calls(ax, genic, chrom, span, color_strand=True)


def plot_tx_transcript_context(outdir, chrom, span, orf_plotter, rna_plotter, combined_model,
	caller, figsize=(12, 5)):

	orf_plotter.set_chrom_span(chrom, span)
	rna_plotter.set_chrom_span(chrom, span, 1)
	combined_model.load_mnase_span(chrom, span)

	plt.figure(figsize=figsize)

	plt.subplot(2, 1, 1)
	ax = plt.gca()
	orf_plotter.plot_orf_annotations(ax)

	plt.subplot(2, 1, 2)
	ax = plt.gca()
	rna_plotter.ylim = 6
	rna_plotter.plot_pileup(ax=ax, mode='minmax')
	ax.set_yticks([])
	ax.set_xticks([])

	caller.plot_called_watson_crick_transcripts_ax(ax)
