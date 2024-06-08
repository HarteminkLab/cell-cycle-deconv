
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.mnase_replication_timing_analysis import get_bin_for_position
from src.geneset import get_deconvolved_geneset
from src.helpers import calcH_config


def plot_reads_bar(corrected_reads, scale=100, color='gray'):

	def plot_row(corrected_reads, row):
		dat = corrected_reads[row, :]
		dat = np.concatenate([dat[0:], dat[-1:]])
		xs = np.arange(len(dat))

		y_offset = row * 2*scale
		y_offset_arr = np.zeros_like(dat)+y_offset

		plt.fill_between(xs, dat+y_offset, y_offset, step='post', color=color, lw=0)
		plt.axhline(y_offset, c='black', lw=0.75)

	n = corrected_reads.shape[0]
	m = corrected_reads.shape[1]

	for i in range(n):
		plot_row(corrected_reads, i)

	spacing_between_plots = scale*2
	yticks = np.arange(spacing_between_plots/2., spacing_between_plots*n, spacing_between_plots)
	yticklabels = ["${t_"+str(i+1)+"}$" for i in range(n)]

	plt.yticks(yticks, yticklabels)

	xticks = np.arange(0, m)
	xticklabels = xticks
	plt.ylim(-spacing_between_plots*0.25, n*spacing_between_plots)


def plot_observed_vs_corrected(rep_profile, observed_reads, corrected_reads):
	plt.figure(figsize=(11, 6))
	plt.subplots_adjust(hspace=0.5)

	plt.subplot(2, 3, 1)
	plot_reads_bar(rep_profile, scale=2., color=plt.get_cmap('tab10')(0))
	plt.title("Replication profile")
	plt.xlabel("Genomic position")

	plt.subplot(2, 3, 2)
	plot_reads_bar(observed_reads)
	plt.title("Observed reads")
	plt.xlabel("Genomic position")

	plt.subplot(2, 3, 3)
	plot_reads_bar(corrected_reads)
	plt.title("Corrected reads")
	plt.xlabel("Genomic position")

	plt.subplot(2, 3, 4)
	plt.plot(rep_profile.mean(axis=1))
	plt.title("Average copies per genome")
	plt.xlabel("Time")

	plt.subplot(2, 3, 5)
	plt.plot(observed_reads.mean(axis=1))
	plt.title("Observed, total reads per time")
	plt.xlabel("Time")

	plt.subplot(2, 3, 6)
	plt.plot(corrected_reads.mean(axis=1))
	plt.title("Corrected, total reads per time")
	plt.xlabel("Time")
