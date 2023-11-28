
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DeconvolvedAnalysis:

	def __init__(self, model):
		"""

		For analysis we will want the g values, f values, H, (TODO: gamma values), interval defintions
		the g values live inside the model config.

		Have an ability to plot a single gene's profile (raw data, and deconvolved data), let's do this first
		Then, we will calculate the peak to trough ratios for every gene

		"""

		f_path = 'output/xin-deconvolved_fs.csv'

		self.model = model

		f = pd.read_csv(f_path)
		f.index = self.model.config.wt1_df.index
		self.f = f


	def plot_deconvolved_gene(self, gene_name, title=None):

		# If I want to plot the initial branch,
		# I need the branch name: i
		# the phases:   R, CG1, postG1
		# and their associated timepoints and indices:
			# Indices is done
			# timepoints are looked up in the intervals object

		from cc_src.sgd import get_orfname

		if gene_name in self.f.index.values:
			orfname = gene_name
		else:
			orfname = get_orfname(gene_name)

		g1 = self.model.config.wt1_df.loc[orfname].values
		g2 = self.model.config.wt2_df.loc[orfname].values

		g = np.concatenate([g1, g2])
		f = self.f.loc[orfname].values

		predicted_g = np.matmul(self.model.H, f)

		predicted_g1 = predicted_g[range(len(g1))]
		predicted_g2 = predicted_g[len(g1):]

		fig, axs = plt.subplots(2, 4, figsize=(16, 7))
		(ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = np.array(axs).flatten()

		# -----------------

		timepoints1 = self.model.config.WT1_TIMEPOINTS
		ax0.plot(timepoints1, g1)
		ax0.plot(timepoints1, predicted_g1)

		# -----------------

		timepoints2 = self.model.config.WT2_TIMEPOINTS
		ax4.plot(timepoints2, g2)
		ax4.plot(timepoints2, predicted_g2)

		# -----------------






