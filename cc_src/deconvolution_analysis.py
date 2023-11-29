
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

		def _plot_branch(ax, branch, start_offset=0, linestyle='solid'):
			"""Plot the branch coloring the individual phases within the branch"""
			phase_tp_idx_list = self.model.config.get_timepoints_phases_Hpositions_for_branch(branch)


			offset = 0
			if start_offset:
				offset = start_offset-phase_tp_idx_list[0][1].values[0]

			for phase, timepoints, indices in phase_tp_idx_list:
				ax.plot(timepoints+offset, f[indices], color=self.color_for_key(phase), 
					lw=5, linestyle=linestyle)

			# return timepoints in case we want to append more branches on to the plot
			return timepoints.values

		# ------------------

		for phase, indices in self.model.config.phase_columns.items():
			ax1.plot(indices, f[indices], color=self.color_for_key(phase), lw=5)
		ax1.set_title("Deconvolved, f")

		_plot_branch(ax2, 't')
		ax2.set_title("Top branch")

		_plot_branch(ax3, 'b')
		ax3.set_title("Bottom branch")

		ax5.imshow(self.model.H, aspect='auto')
		ax5.set_title('Convolution kernel, H')

		_plot_branch(ax6, 'i')
		ax6.set_title("Initial branch")

		i_timepoints = _plot_branch(ax7, 'i', linestyle='dashed')
		_plot_branch(ax7, 'b', start_offset=i_timepoints[-1], linestyle='dashed')
		ax7.set_title("Single cell profile")



	def color_for_key(self, key):
		"""Predefined colors for phases and keys for gene plots"""

		color_map = {
			 "raw": np.array([158, 50, 50])/255.,
			 "fit": np.array([145, 180, 98])/255.,
			 "R": np.array([199, 148, 144])/255.,
			 "RG1": np.array([199, 148, 144])/255.,
			 "CG1": np.array([147, 168, 198])/255.,
			 "DG1": np.array([165, 197, 204])/255.,
			 "postG1": np.array([223, 192, 158])/255.,
			 "H": np.array([100, 100, 100])/255.
		}

		return color_map[key]




