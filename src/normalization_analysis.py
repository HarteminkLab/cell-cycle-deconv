
# Preamble, notebook setup and imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NormAnalysis:

	def __init__(self):

		from src.config import load_xg_gene_expression_config, read_yl_vst_data_rep

		self.yl_rep1_TPM = pd.read_csv('datasets/yl_cell_cycle/replicate1_gene_expression_TPM.csv'
			).set_index('orf_name')

		self.yl_rep1_vst = read_yl_vst_data_rep(1)

		# The Xin/Orlando dataset is a little more hairy than a csv with labeled rows and columns
		# so let's load with the config
		self.xg_ge_wt1 = load_xg_gene_expression_config().wt1_df

	def compute_meta(self):
		self.vst_diff = compute_differences(self.yl_rep1_vst)
		self.tpm_diff = compute_differences(self.yl_rep1_TPM)
		self.xg_diff = compute_differences(self.xg_ge_wt1)

	def plot_differences(self):

		highlight_orfs = ['YER070W']

		vst_diff = self.vst_diff
		tpm_diff = self.tpm_diff
		xg_diff = self.xg_diff

		plt.figure(figsize=(15, 9))
		plt.subplots_adjust(hspace=0.3, wspace=0.3)

		plt.subplot(2, 3, 1)
		plt.scatter(xg_diff['min'], xg_diff['max'], s=1, alpha=0.2, c='green')
		selected_dat = xg_diff.loc[highlight_orfs]
		plt.scatter(selected_dat['min'], selected_dat['max'], s=10, alpha=1.0, c='black')	

		plt.plot([0, 30000], [0, 30000], c='black', lw=1, linestyle='dotted')
		plt.title('Microarrays')
		plt.ylabel("max")
		plt.xlabel("min")

		plt.subplot(2, 3, 2)
		plt.scatter(tpm_diff['min'], tpm_diff['max'], s=1, alpha=0.2)
		selected_dat = tpm_diff.loc[highlight_orfs]
		plt.scatter(selected_dat['min'], selected_dat['max'], s=10, alpha=1.0, c='black')
		plt.plot([0, 10000], [0, 10000], c='black', lw=1, linestyle='dotted')
		plt.xlim(-500, 7500)
		plt.ylim(-500, 7500)
		plt.title('RNA-seq - TPM')
		plt.ylabel("max")
		plt.xlabel("min")

		plt.subplot(2, 3, 3)
		plt.scatter(vst_diff['min'], vst_diff['max'], s=1, alpha=0.2, c='orange')

		selected_dat = vst_diff.loc[highlight_orfs]
		plt.scatter(selected_dat['min'], selected_dat['max'], s=10, alpha=1.0, c='black')

		plt.title('RNA-seq - VST')
		plt.ylabel("max")
		plt.xlabel("min")

		plt.subplot(2, 3, 4)
		plt.scatter(xg_diff['min'], xg_diff['max'], s=1, alpha=0.2, c='green')

		selected_dat = tpm_diff.loc[highlight_orfs]
		plt.scatter(selected_dat['min'], selected_dat['max'], s=10, alpha=1.0, c='black')

		plt.title('Microarrays')
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel("log max")
		plt.xlabel("log min")

		plt.subplot(2, 3, 5)
		plt.scatter(tpm_diff['min']+1, tpm_diff['max']+1, s=1, alpha=0.2)

		selected_dat = tpm_diff.loc[highlight_orfs]
		plt.scatter(selected_dat['min'], selected_dat['max'], s=10, alpha=1.0, c='black')

		plt.plot([0, 10000], [0, 10000], c='black', lw=1, linestyle='dotted')
		plt.title('RNA-seq - TPM')
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel("log max")
		plt.xlabel("log min")

		plt.subplot(2, 3, 6)
		plt.scatter(vst_diff['min'], vst_diff['max'], s=1, alpha=0.2, c='orange')

		selected_dat = vst_diff.loc[highlight_orfs]
		plt.scatter(selected_dat['min'], selected_dat['max'], s=10, alpha=1.0, c='black')

		plt.plot([0, 20], [0, 20], c='black', lw=1, linestyle='dotted')
		plt.title('RNA-seq - VST')
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel("log max")
		plt.xlabel("log min")


def compute_differences(df):
	diff_df = df[[]].copy()
	diff_df['min'] = df.min(axis=1)
	diff_df['max'] = df.max(axis=1)
	diff_df['difference'] = diff_df['max'] - diff_df['min']

	return diff_df
