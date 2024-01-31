
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SpellmanAnalysis:
	"""Analysis of the spellman genes for the yulong replicates"""

	def __init__(self):

		gene_as_table = pd.read_csv('datasets/datasets_from_web_deconvolution.cs.duke.edu/gene_associated.tsv', sep='\t')
		spellman_genes = gene_as_table[gene_as_table['Spellman1998'] == '1']
		spellman_orfs = spellman_genes['Systematic Name'].values
		print(f"There are {len(spellman_genes)} Spellman genes.")
		self.spellman_genes = spellman_genes

		# Load the two wildtype datasets
		from src.config import load_combined_yl_vst_gene_expression_config
		config = load_combined_yl_vst_gene_expression_config()
		self.wt1 = config.wt1_df
		self.wt2 = config.wt2_df

		self.wt1_spellman_normalized = normalize_genes(self.wt1.loc[spellman_orfs])
		self.wt2_spellman_normalized = normalize_genes(self.wt2.loc[spellman_orfs])


	def plot_cluster_maxes(self, replicate):

		plt.figure(figsize=(8, 8))
		plt.subplots_adjust(top=0.9, hspace=0.4)

		if replicate == 1:
			plot_data = self.wt1_spellman_normalized.copy()
			plot_data['cluster'] = self.wt1_spellman_cluster
			times = self.wt1.columns
			plt.suptitle(f"Replicate 1")
		else:
			plot_data = self.wt2_spellman_normalized.copy()
			plot_data['cluster'] = self.wt2_spellman_cluster
			times = self.wt2.columns
			plt.suptitle(f"Replicate 2")

		import matplotlib.patheffects as path_effects
		for cluster in range(self.k):
			
			plt.subplot(3, 3, cluster+1)
			cluster_data = plot_data[plot_data['cluster'] == cluster]
			mean_data = cluster_data.mean(axis=0)
			
			for orf_name, row in cluster_data.iterrows():
				plt.plot(times, row[times], c='red', alpha=0.05)
			plt.plot(times, mean_data[times], c='blue')
			plt.title(f"Clust {cluster+1}, n={len(cluster_data)}")

			maxes = get_maxes(cluster_data[times])
			max_times = times[maxes]

			for m in max_times:
				plt.axvline(m, c='black')
				text = plt.text(m, 2, f"{m}", ha='center')
				text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
								   	   path_effects.Normal()])

	def cluster(self, k):
		from sklearn.cluster import KMeans

		self.k = k
		kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(self.wt1_spellman_normalized)
		self.wt1_spellman_cluster = kmeans.labels_

		kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(self.wt2_spellman_normalized)
		self.wt2_spellman_cluster = kmeans.labels_

	def plot_data(self):

		fig = plt.figure(figsize=(8, 6))
		plt.suptitle(f"Spellman genes, n={len(self.wt1_spellman_normalized)}")
		plt.subplots_adjust(top=0.85, hspace=0.3)

		def plot_spellman(data):
			for orf_name, row in data.iterrows():
				vals = row.values
				plt.plot(data.columns, vals, lw=1, alpha=0.02, c='red') 

		plt.subplot(2, 2, 1)
		plt.title("Replicate 1, normalized")
		plot_spellman(self.wt1_spellman_normalized)
		plt.subplot(2, 2, 3)
		plt.title("Replicate 1, mean")
		plt.plot(self.wt1_spellman_normalized.mean(axis=0))
			
		plt.subplot(2, 2, 2)
		plot_spellman(self.wt2_spellman_normalized)
		plt.title("Replicate 2, normalized")
		plt.subplot(2, 2, 4)
		plt.plot(self.wt2_spellman_normalized.mean(axis=0))
		plt.title("Replicate 2, mean")


def normalize_genes(data):
	"""z-score normalize"""
	normalized_data = data.copy()
	for orf_name, row in data.iterrows():
		vals = row.values
		mean, sd = vals.mean(), vals.std()
		vals = (vals - mean)/(sd+1e-4)
		normalized_data.loc[orf_name] = vals
	return normalized_data


def get_maxes(data):
	"""
	Get the three highest peaks of the data by the mean
	"""
	def neginf_around(data, idxmax):
		
		padding = 3
		updated_arr = data.copy()
		start = max(0, idxmax-padding)
		end = min(idxmax+padding, len(updated_arr))
		updated_arr[start:end] = float('-inf')
		return updated_arr

	mean_data = data.mean(axis=0).values

	# First max
	idxmax = mean_data.argmax()
	updated_arr = neginf_around(mean_data, idxmax)

	# Second max
	idxmax2 = updated_arr.argmax()
	updated_arr = neginf_around(updated_arr, idxmax2)

	idxmax3 = updated_arr.argmax()

	return sorted([idxmax, idxmax2, idxmax3])
