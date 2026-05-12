import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pipeline.cluster_metrics_data import load_from_chromatin_and_expression_processor
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper

class ChromatinMetrics4D(object):
	"""This class allows us to analyze each of the chromatin metrics
	and expression against one another and together.

	This will be done in both: (1) the PTR space and (2) the trajectory space
	for each gene.
	"""
	def __init__(self, fig_metrics):
		super(ChromatinMetrics4D, self).__init__()

		self.fig_metrics = fig_metrics

		# Create the 4d trajectory data
		self.metrics_gene_data = load_from_chromatin_and_expression_processor(
			fig_metrics.chromatin_processor,
			fig_metrics.expression_processor)

		# Create the 4d metrics data
		self.joined_metrics_ptr_data = self.create_joined_ptrs_dataset(fig_metrics)

		# Make directory for saving
		mkdir_safe(self.save_dir)

	@property
	def save_dir(self) -> str:
		return f"{self.fig_metrics.output_dir}/chromatin_metrics/clustering"

	def cluster_trajectories(self):
		# Cluster using Kmeans

		from sklearn.preprocessing import StandardScaler
		from sklearn.cluster import KMeans

		# Z-normalize
		scaler = StandardScaler()
		data_normalized = scaler.fit_transform(self.metrics_gene_data.data_2d)

		# Try K-means
		k = 6
		kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
		clusters = kmeans.fit_predict(data_normalized)

		self.trajectory_k = k
		self.trajectory_clusters = clusters

	def plot_trajectories_PaCMAP(self, data=None, clusters=None):
		import pacmap

		# Plot using PaCMAP
		# initializing the pacmap instance
		# Setting n_neighbors to "None" leads to an automatic choice 
		# shown below in "parameter" section
		embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, 
									  MN_ratio=0.5, FP_ratio=1.0)

		if data is None:
			data = self.filtered_data_normalized_2d
			clusters = self.clusters

		# fit the data (The index of transformed data corresponds to the 
		# index of the original data)
		X_transformed = embedding.fit_transform(data, init="pca")

		if clusters is None:
			clusters = self.trajectory_clusters

		# Visualize the embedding
		fig, ax = plt.subplots(1, 1, figsize=(6, 4))

		scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=12, 
			edgecolor='#aaa', facecolor='none')
		scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=10, 
			c=clusters,
			cmap='Spectral_r')
		# plt.xlim(-14, 14)
		# plt.ylim(-6, 6.75)
		plt.title(f"PaCMAP of chromatin-expression\ntrajectory space, n={len(X_transformed)}", 
			fontweight='demi')


		# Add legend
		legend1 = ax.legend(*scatter.legend_elements(),
							title="Clusters",
							loc="best")
		ax.add_artist(legend1)
		save_figure_for_paper('clusters_PaCMAP.png')

	def plot_cluster_trajectories(self):
		from pipeline.cluster_metrics_data_helpers import plot_pairs_index
		from src.utils import mkdir_safe
		from src.figure_configs import save_figure_for_paper

		save_dir = f"{self.save_dir}/trajectories"
		mkdir_safe(save_dir)

		num_examples = 4
		clusters = self.clusters
		data_normalized_3d = self.filtered_data_normalized_3d
		optimal_k = self.optimal_k

		for current_k in [1, 11, 13]:#range(1, optimal_k+1):

			print(f"Cluster {current_k}")

			# Get the gene array indices for the genes in the current cluster
			# and the associated orf names
			cluster_gene_indices = np.where(clusters == current_k)[0]
			
			masked_orfnames = self.filtered_set_orfnames
			cluster_orf_names = masked_orfnames[cluster_gene_indices]
			
			# The average gene in the cluster
			cluster_meta_gene = np.mean(data_normalized_3d[cluster_gene_indices], axis=0)

			color = plt.cm.Spectral_r(current_k/optimal_k)
			example_indices = np.where(clusters == current_k)[0]
				
			orf_names_to_plot = masked_orfnames[example_indices]
			
			# for i in range(num_examples):
				
			# 	try:
			# 		example_i = example_indices[i]
			# 		orf_name = orf_names_to_plot[i]
			# 		from src.sgd import get_gene_name
			# 		gene_name = get_gene_name(orf_name)
					
			# 		example_data = data_normalized_3d[example_i]
			# 		plot_pairs_index(example_data, color=color, expanded_lims=True)
			# 		plt.suptitle(f"Cluster {current_k}, example {example_i}, {gene_name}")
			# 		save_figure_for_paper(f"{save_dir}/cluster_{current_k}_{example_i}.png")

			# 	except Exception:
			# 		print(f"Error with cluster {current_k}, example {example_i}, skipping")
			# 		continue

			# 	plt.cla()
			# 	plt.clf()

			title = f"Cluster {current_k} mean, n={len(cluster_gene_indices)}"

			if current_k == 11:
				title = "Chromatin reorganization\n"\
						f"Cluster 11 mean, n={len(cluster_gene_indices)}"
			elif current_k == 13:
				title = "DNA replication and repair\n"\
					f"Cluster 13 mean, n={len(cluster_gene_indices)}"

			plot_pairs_index(cluster_meta_gene)
			plt.suptitle(title,
				fontsize=20, fontweight='demi')
			save_figure_for_paper(f"{save_dir}/meta_cluster_{current_k}.png")

			# plt.cla()
			# plt.clf()
				
			# plt.close('all')
			# break

	def run_go_clusters(self):
		from src.gene_ontology import GeneOntology

		gene_ontology = GeneOntology()
		from src.sgd import read_nondubious_genes_dataset

		masked_set_of_genes = self.filtered_set_orfnames
		clusters = self.clusters

		all_go_results_df = pd.DataFrame()
		for current_cluster in range(1, self.optimal_k+1):

			genes = read_nondubious_genes_dataset()

			cluster_gene_indices = np.where(clusters == current_cluster)[0]
			selected_orfs = masked_set_of_genes[cluster_gene_indices]

			selected_gene_names = genes.loc[selected_orfs]['gene'].values
			gene_ontology.run_go(selected_gene_names)
			results = gene_ontology.results_df.copy()
			results['cluster'] = current_cluster

			all_go_results_df = pd.concat([all_go_results_df, results])

		results = all_go_results_df

		omit_groups = ['biological_process', 'cellular_component',
			'molecular_function', 'plasma membrane', 'extracellular region',
			'nucleus']
		all_results = results[~results['name'].isin(omit_groups)].set_index(['cluster'])
		sig_results = all_results[all_results.bonferroni < 0.05]

		self.all_results = all_results
		self.sig_results = sig_results


	def plot_pairs_of_ptrs(self, ptrs_data, mask_filter=None):

		from pipeline.cluster_metrics_data_helpers import generate_metrics_comparison_plot
		fig, triangle_axes = generate_metrics_comparison_plot(figsize=(6, 6))

		pairs_to_plot = [
			('promoter_occupancy_ptr', 'expression_ptr'),
			('nucleosome_occupancy_ptr', 'expression_ptr'),
			('nucleosome_entropy_ptr', 'expression_ptr'),
			('promoter_occupancy_ptr', 'nucleosome_entropy_ptr'),
			('nucleosome_occupancy_ptr', 'nucleosome_entropy_ptr'),
			('promoter_occupancy_ptr', 'nucleosome_occupancy_ptr'),
		]

		plt.figure(figsize=(9, 6))

		for i, pair in enumerate(pairs_to_plot):
			ax = triangle_axes[i]

			masked_data = ptrs_data[mask_filter]
			non_masked_data = ptrs_data[~mask_filter]

			ax.scatter(non_masked_data[pair[0]], non_masked_data[pair[1]], s=1,
					   c='gray')

			ax.scatter(masked_data[pair[0]], masked_data[pair[1]], s=1,
					   c='red', label="Selected genes (>90th percentile)")

			ax.set_xlim(0.95, 3)
			ax.set_ylim(0.95, 3)

			def _convert_metric_name_to_title(metric_name):
				title = metric_name.replace('_ptr', '').replace('_', '\n')
				title = title[0].upper() + title[1:]
				return title

			if i < 3:	
				ax.set_title(_convert_metric_name_to_title(pair[0]), 
					fontsize=13)

			if i in [0, 3, 5]:
				ax.set_ylabel(_convert_metric_name_to_title(pair[1]),
					fontsize=13)

		plt.subplots_adjust(hspace=0.35, wspace=0.35)


	def threshold_metrics_by_diameter(self, q_threshold=0.9):
		from pipeline.cluster_metrics_data_helpers import compute_diameter

		self.normalize_data()

		# Zero out the transcription dimension
		#chromatin_only_data_normalized_3d = self.data_normalized_3d
		#chromatin_only_data_normalized_3d[:, 3, :] = 0

		self.trajectory_diameters = compute_diameter(self.data_normalized_3d)#chromatin_only_data_normalized_3d)
		diameters = self.trajectory_diameters

		diameter_threshold = np.quantile(diameters, q=q_threshold)
		mask_filter = diameters > diameter_threshold
		masked_set_of_genes = self.metrics_gene_data.gene_row_index[mask_filter]

		# Examine how diameter filtering will look when plotting the PTRs

		plt.figure(figsize=(4, 3))
		plt.hist(diameters, bins=200)
		plt.axvline(diameter_threshold, c='red')
		plt.xlim(0, 10)
		plt.title("Distribution of trajectory diameters", fontweight='demi',
			fontsize=16)
		plt.xlabel("Trajectory diameter", fontsize=14)
		save_figure_for_paper(f"{self.save_dir}/diameters_hist.png")

		self.plot_pairs_of_ptrs(self.joined_metrics_ptr_data, mask_filter)
		plt.suptitle(f"Filtered vs unfiltered genes by diameter threshold\n"
					 f"n={mask_filter.sum()} / {len(diameters)}")

		self.diameter_threshold = diameter_threshold
		self.filtered_set_orfnames = masked_set_of_genes
		self.create_set_of_masked_genes()
		self.filtered_data_normalized_3d = self.data_normalized_3d[self.array_indices_of_filtered_set]
		self.filtered_data_normalized_2d = self.data_normalized_2d[self.array_indices_of_filtered_set]
		save_figure_for_paper(f"{self.save_dir}/ptrs_diameter_cutoff.png")


	def create_set_of_masked_genes(self):
		"""Using the diameter thresholding criteria, create a set for the
		filtered data and gene orfnames for clustering and downstream
		analysis"""

		# Get the indices of the filtered genes
		all_genes_index = self.metrics_gene_data.gene_row_index
		self.array_indices_of_filtered_set = np.where(np.isin(all_genes_index, 
			self.filtered_set_orfnames))[0]


	def normalize_data(self):
		# Cluster the trajectories
		from sklearn.preprocessing import StandardScaler
		from sklearn.cluster import KMeans

		data_2d = self.metrics_gene_data.data_2d
		data_3d = self.metrics_gene_data.data_3d

		# Calculate mean and std for each metric (axis=(0,2) averages across samples and timepoints)
		means = np.mean(data_3d, axis=(0, 2), keepdims=True)  # Shape: (1, 4, 1)
		stds = np.std(data_3d, axis=(0, 2), keepdims=True)    # Shape: (1, 4, 1)

		# Normalize along axes
		data_normalized_3d = (data_3d - means) / stds

		# Reshape to the flattened shape
		data_normalized_2d = data_normalized_3d.reshape(data_2d.shape)
		self.data_normalized_3d = data_normalized_3d
		self.data_normalized_2d = data_normalized_2d


	def kmeans_determine_k(self, chromatin_only=True):

		# Elbow method along a wide range of k values
		K_range = range(8, 60)
		inertias = []

		for k in K_range:
			kmeans = self.kmeans_cluster(k, chromatin_only)
			inertias.append(kmeans.inertia_)

		# Plot
		plt.figure(figsize=(4, 3))
		plt.plot(K_range, inertias)
		plt.xlabel('Number of Clusters (k)', fontweight=12)
		plt.ylabel('Inertia (WCSS)', fontweight=12)
		plt.title('Elbow Method for Optimal k', fontweight='demi',
			fontsize=15)

		from kneed import KneeLocator
		kn = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
		optimal_k = K_range[kn.maxima_indices[0]]
		plt.axvline(optimal_k, c='red', label=f"k={optimal_k:.0f}")
		print("The optimal_ke optimal k for clustering is: ", optimal_k)
		plt.legend()
		save_figure_for_paper('k_elbow.png')

		self.optimal_k = optimal_k

	def kmeans_cluster(self, k=None, chromatin_only=True):

		from sklearn.cluster import KMeans

		if k is None:
			k = self.optimal_k

		if chromatin_only:
			# If we were to cluster only on chromatin data, 
			# we can zero out the transcription dimension
			chromatin_only_data_normalized_3d = self.filtered_data_normalized_3d.copy()
			chromatin_only_data_normalized_3d[:, 3, :] = 0
			chromatin_only_data_normalized_2d = chromatin_only_data_normalized_3d\
				.reshape(self.filtered_data_normalized_2d.shape)
			data_to_cluster = chromatin_only_data_normalized_2d
		else:
			data_to_cluster = self.filtered_data_normalized_2d

		kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
		self.kmeans = kmeans.fit(data_to_cluster)
		self.clusters = self.kmeans.labels_ + 1 # Index cluster names by 1
		return kmeans


	def create_cluster_summaries(self):

		def create_cluster_counts(clusters):
			cluster_names, counts = np.unique(clusters, return_counts=True)
			cluster_counts_df = pd.DataFrame(counts, index=cluster_names, columns=['count'])
			cluster_counts_df['cumulative_sum'] = cluster_counts_df.cumsum()
			cluster_counts_df['label_position'] = cluster_counts_df['cumulative_sum']\
				- cluster_counts_df['count'] / 2
			return cluster_counts_df

		cluster_counts_df = create_cluster_counts(self.clusters)
		self.cluster_counts = cluster_counts_df

		from pipeline.cluster_metrics_data_helpers import test_tf_enrichment_fisher

		tf_counts = self.filtered_cluster_tf_summary.join(self.cluster_counts[['count']], how='right').fillna(0)
		tf_counts.loc[:] = tf_counts.values.astype(int)
		self.tf_binding_adj_p_values = test_tf_enrichment_fisher(tf_counts).pivot(
			index='cluster', columns='TF', values='adjusted_p_value').fillna(1)


	def plot_heatmap_of_clusters(self):

		from src.figure_configs import tf_colors
		self.create_cluster_summaries()

		clusters = self.clusters
		data_normalized_2d = self.filtered_data_normalized_2d

		enriched_terms = self.sig_results
		cluster_counts_df = self.cluster_counts
		self.cluster_counts = cluster_counts_df

		sorted_indices = np.argsort(clusters)

		fig, (ax_chromatin, ax_expression, ax_tfs) = plt.subplots(1, 3, figsize=(13, 10), 
									 gridspec_kw={'width_ratios': [0.45, 0.15, 0.25]})
		# ax_chromatin = chromatin heatmap (3 metrics wide)
		# ax_expression = expression heatmap (1 metric wide)  
		# ax_tf3 = TF binding counts

		len_of_tps = 128
		chromatin_data = data_normalized_2d[sorted_indices, :len_of_tps*3]  # First 384 columns
		expression_data = data_normalized_2d[sorted_indices, len_of_tps*3:]  # Last 128 columns

		# Plot chromatin heatmap
		im = ax_chromatin.imshow(chromatin_data, vmin=-4, vmax=4, 
						cmap='RdYlBu_r',
						aspect='auto', interpolation='none')

		# Plot expression heatmap
		im = ax_expression.imshow(expression_data, vmin=-4, vmax=4, 
						cmap='RdYlBu_r',
						aspect='auto', interpolation='none')

		from src.config import retrieve_phase_index_ticks, load_default_chrom_configs
		config1, _ = load_default_chrom_configs()
		phase_ticks, edge_ticks, tick_labels = retrieve_phase_index_ticks('t', 
			config1, with_labels=True)

		edge_ticks = edge_ticks[:-1] + [edge_ticks[-1]+1]
		# Calculate the period length (distance between first and last edge tick)
		period_length = edge_ticks[-1] - edge_ticks[0]

		# Create repeated ticks by appending offset versions
		all_phase_ticks = phase_ticks.copy()
		all_tick_labels = tick_labels.copy()
		all_edge_ticks = edge_ticks.copy()

		ax_expression.set_xticks(phase_ticks)
		ax_expression.set_xticklabels(tick_labels, fontsize=10)
		ax_expression.set_xticks(edge_ticks, minor=True)
		ax_expression.set_yticks([])

		for i in range(1, 3):  # Repeat 2 more times
			offset = period_length * i
			# Append phase ticks (excluding the first one to avoid duplication at boundaries)
			all_phase_ticks = np.concatenate([all_phase_ticks, phase_ticks + offset])
			all_edge_ticks = np.concatenate([all_edge_ticks, edge_ticks + offset])
			all_tick_labels = all_tick_labels + tick_labels

		ax_chromatin.set_xticks(all_phase_ticks)
		ax_chromatin.set_xticklabels(all_tick_labels, fontsize=10)
		ax_chromatin.set_xticks(all_edge_ticks, minor=True)

		ax_chromatin.tick_params(axis='x', which='major', length=0)
		ax_chromatin.tick_params(axis='x', which='minor', length=10) 
		ax_expression.tick_params(axis='x', which='major', length=0)
		ax_expression.tick_params(axis='x', which='minor', length=10) 

		# Load the TF summary data
		tf_summary = self.filtered_cluster_tf_summary
		def _sort_tf_summary_table_by_groupings(tf_summary):
			# Get the ordered TF names from the dictionary
			tf_order = list(tf_colors.keys())
			df = tf_summary
			# Filter to only include columns that exist in the dataframe
			ordered_cols = [col for col in tf_order if col in df.columns]
			# Get any remaining columns not in tf_colors (to preserve them)
			remaining_cols = [col for col in df.columns if col not in tf_order]
			# Reorder the dataframe
			df = df[ordered_cols + remaining_cols]
			return df
		tf_summary = _sort_tf_summary_table_by_groupings(tf_summary)

		# Plot the vertical grid lines for the TFs plot
		tf_ticks = np.arange(len(tf_summary.columns))
		for i, x in enumerate(tf_ticks):
			tf_name = tf_summary.columns[i]
			color = tf_colors.get(tf_name, tf_colors['other'])
			ax_tfs.axvline(x, c=color, lw=0.5, zorder=0, ls=(1, (1, 1)))

		# Plot the horizontal cluster separators
		for cluster_name, row in cluster_counts_df.iterrows():
			ax_chromatin.axhline(row.cumulative_sum, c='black', lw=1.5)
			ax_tfs.axhline(row.cumulative_sum, c='black', lw=1.5)
			ax_expression.axhline(row.cumulative_sum, c='black', lw=1.5)
		ax_chromatin.set_ylim(row.cumulative_sum, 0)
		ax_expression.set_ylim(row.cumulative_sum, 0)
		ax_tfs.set_ylim(row.cumulative_sum, 0)
		ax_chromatin.set_ylabel("Cluster", fontsize=18, fontweight='demi')
		ax_tfs.set_yticks([])

		yticks = []
		ytick_labels = []

		right_yticks = []
		right_ytick_terms = []

		import matplotlib.patheffects as path_effects

		for cluster, row in cluster_counts_df.iterrows():

			# If a cluster has more than 5 genes
			if row['count'] > 5:

				# Label cluster
				ytick = row.label_position
				ytick_label = f"{row.name}, n={row['count']:.0f}"
				yticks.append(ytick)
				ytick_labels.append(ytick_label)

				# Label GO terms
				if cluster in enriched_terms.index:

					cluster_terms = enriched_terms.loc[cluster]
					if type(cluster_terms) == pd.core.series.Series:
						cluster_terms = [enriched_terms.loc[cluster]['name']]
					else:            
						cluster_terms = enriched_terms.loc[cluster]['name'].values

					cluster_terms = [t if 'mRNA' in t else t[0].upper() + t[1:] for t in cluster_terms]
					terms = '\n'.join(cluster_terms)

					right_yticks.append(ytick)
					right_ytick_terms.append(terms)

				# Cluster TFs
				if cluster in tf_summary.index:
					tfs_in_cluster = tf_summary.loc[cluster]

					for i, tf in enumerate(tfs_in_cluster.index):
						count = tfs_in_cluster.loc[tf]
						if count >= 1:

							x = tf_ticks[i]
							y = row.label_position

							p_value = self.tf_binding_adj_p_values.loc[cluster][tf]

							sig = ""
							if p_value < 0.001:
								sig = '**'
							elif p_value < 0.05:
								sig = '*'

							color = tf_colors.get(tf, tf_colors['other'])

							plt.text(x, y, f"{count}$^" + "{" + sig + "}$", color=color,
								ha='center', va='center',
								fontweight='bold', fontsize=14,
								path_effects=[path_effects.withStroke(linewidth=2, 
									foreground='white')])

		ax_chromatin.set_yticks(yticks, ytick_labels, fontsize=16, fontweight='demi')
		len_of_tps = 128
		len_tps_2 = len_of_tps//2

		metric_titles = []

		for title in self.metrics_gene_data.metric_names:
			modified_title = (title[0].upper() + title[1:]).replace('_', '\n')
			metric_titles.append(modified_title)

		top_ax_chromatin = ax_chromatin.twiny()
		top_ax_chromatin.set_xlim(ax_chromatin.get_xlim())
		top_ax_chromatin.set_xticks(np.arange(len_tps_2, len_of_tps*3, len_of_tps))
		top_ax_chromatin.set_xticklabels(metric_titles[:3], rotation=0, fontweight='demi',
					ha='center', fontsize=14)

		ax_expression.set_title('Expression', fontsize=14, fontweight='demi')

		ax_right = ax_tfs.twinx()
		ax_right.set_ylim(ax_chromatin.get_ylim())
		ax_right.set_yticks(right_yticks)
		ax_right.set_yticklabels(right_ytick_terms, fontsize=13)
		ax_right.tick_params(axis='x', which='major', length=0)
		top_ax_chromatin.tick_params(axis='x', which='major', length=0)

		for col, x in enumerate(range(len_of_tps, len_of_tps*4, len_of_tps)):
			if col == 2:
				lw = 1.5
				ls = 'solid'
			else:
				lw = 1
				ls = 'dotted'

			ax_chromatin.axvline(x-0.5, c='black', lw=lw, ls=ls)

		ax_tfs.set_xticks(tf_ticks)
		ax_tfs.set_xticklabels(tf_summary.columns, rotation=90, fontsize=11,
			fontweight='demi')
		ax_tfs.set_xlim(tf_ticks[0]-0.5, tf_ticks[-1]+0.5)
		ax_tfs.set_title("TF binding", fontweight='demi')

		# Add colored borders to each tick label
		for i, (tick_label, tf_name) in enumerate(zip(ax_tfs.get_xticklabels(),
			tf_summary.columns)):
			color = tf_colors.get(tf_name, tf_colors['other'])
			tick_label.set_color('white')
			tick_label.set_bbox(dict(
				boxstyle='round,pad=0.2',
				facecolor=color,
				edgecolor=color,
				linewidth=1
			))

		def format_axes(ax, spine_border_width=1.5):
			ax.spines['top'].set_linewidth(spine_border_width)
			ax.spines['bottom'].set_linewidth(spine_border_width)
			ax.spines['left'].set_linewidth(spine_border_width)
			ax.spines['right'].set_linewidth(spine_border_width)

		format_axes(ax_chromatin)
		format_axes(ax_expression)
		format_axes(ax_tfs)

		plt.suptitle((f"Genes clustered by chromatin trajectories\n"
					  f"n={len(self.filtered_set_orfnames)} genes, k={self.optimal_k} clusters"), 
			fontweight='demi',
				 fontsize=25,
				 y=1.0)
		plt.tight_layout()
		plt.subplots_adjust(wspace=0.05)

		# Add the colorbar
		cbar_ax = fig.add_axes([0.84, 0.6, 0.02, 0.25])  # [left, bottom, width, height]
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.set_label('Normalized value', rotation=270, labelpad=10)

		save_figure_for_paper(f"{self.save_dir}/clusters_heatmap.png")


	def plot_cluster_mean_3d_plot(self):
		from pipeline.cluster_metrics_data_helpers import plot_meta_gene_3d

		clusters = self.clusters
		data_normalized_3d = self.filtered_data_normalized_3d
		indices_of_masked_genes = self.array_indices_of_filtered_set

		for current_k in range(1, self.optimal_k+1):

			# Get the gene array indices for the genes in the current cluster
			# and the associated orf names
			cluster_gene_indices = np.where(clusters == current_k)[0]
			
			masked_orfnames = self.metrics_gene_data.gene_row_index[indices_of_masked_genes]
			cluster_orf_names = masked_orfnames[cluster_gene_indices]
			
			# The average gene in the cluster
			cluster_meta_gene = np.mean(data_normalized_3d[cluster_gene_indices], axis=0)
			
			#if len(cluster_orf_names) > 20:
			print(f"Cluster {current_k}")
			plot_meta_gene_3d(cluster_meta_gene)
			plt.suptitle(f"Cluster {current_k}, n={len(cluster_orf_names)}", 
						 fontweight='demi', fontsize=16, y=0.85)

			save_figure_for_paper(f"{self.save_dir}/trajectories/3d_meta_cluster_{current_k}.png")
			plt.cla()
			plt.clf()

			
	def plot_trajectories_example(self):
		"""Plot the trajectories for pairs of metrics.

		To do: this code currently plots a cluster example, and can
		be changed to plot various gene examples."""
		from pipeline.cluster_metrics_data_helpers import plot_pairs_index, plot_single_pair

		def plot_example_cluster(cluster_num=None, gene_name=None, 
			x_metric_index=0, y_metric_index=3, xlim=None, ylim=None,
			cell_phase_formatting={}, ax=None):

			# Get the gene array indices for the genes in the current cluster
			# and the associated orf names

			if gene_name is not None:
				from src.sgd import get_orfname
				orf_name = get_orfname(gene_name)
				gene_indices_to_plot = np.where(self.filtered_set_orfnames == orf_name)[0]
			else:
				gene_indices_to_plot = np.where(self.clusters == cluster_num)[0]

			
			masked_orfnames = self.filtered_set_orfnames
			cluster_orf_names = masked_orfnames[gene_indices_to_plot]
			
			# The average gene in the cluster
			data_normalized_3d = self.filtered_data_normalized_3d
			cluster_meta_gene = np.mean(data_normalized_3d[gene_indices_to_plot], axis=0)

			x_values = cluster_meta_gene[x_metric_index]
			y_values = cluster_meta_gene[y_metric_index]

			x_values = x_values - x_values.mean()
			y_values = y_values - y_values.mean()

			metrics = [
				'promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy', 
				'expression']

			plot_single_pair(
				ax=ax,
				data_x=x_values,
				data_y=y_values,
				x_label=metrics[x_metric_index],
				y_label=metrics[y_metric_index],
				xlim=xlim,
				ylim=ylim,
				plot_arrows=True,
			)

			# Plot the gene or cluster name on the diagram
			if gene_name is not None:
				trajectory_label = '$\\it{' + gene_name + '}$'
			else:
				trajectory_label = f'Cluster {cluster_num}'

			from src.plot_helpers import color_for_key

			def _plot_phase_text(index, phase, ha='center', va='center', offset=(0, 0)):
				if phase == 'meanG1':
					text = 'G1'
				else:
					text = phase
				ax.text(x_values[index]+offset[0], y_values[index]+offset[1], 
					text, color=color_for_key(phase),
					ha=ha, va=va, fontweight='demi', fontsize=15)

			for phase, (index, ha, va, offset) in cell_phase_formatting.items():
				_plot_phase_text(index, phase, va=va, ha=ha, offset=offset)

			from src.plot_helpers import create_first_character_title

			def _create_title_from_metric(title):
				return create_first_character_title(title.replace('_', ' '))

			ax.set_xlabel(_create_title_from_metric(metrics[x_metric_index]), fontsize=16)
			ax.set_ylabel(_create_title_from_metric(metrics[y_metric_index]), fontsize=16)

			xlim_padding = (xlim[1]-xlim[0])*0.025
			ylim_padding = (ylim[1]-ylim[0])*0.025

			# Label of the gene or cluster
			ax.text(xlim[1]-xlim_padding, ylim[0]+ylim_padding, 
				trajectory_label, color='black',
				ha='right', va='bottom', fontweight='regular', fontsize=14)

		fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(4, 14))

		cell_phase_formatting = {
						'meanG1': (40, 'left', 'top', (0.01, 0.0)),
						'S': (85, 'left', 'bottom', (0.25, 0.01)),
						'G2/M': (100, 'right', 'center', (-0.1, 0.0))}
		plot_example_cluster(gene_name='HTA2', x_metric_index=0, y_metric_index=3,
				xlim=(-2, 2), ylim=(-2, 2), 
				cell_phase_formatting=cell_phase_formatting, ax=ax0)
		ax0.set_title("Direct, positively\n linked (slope) measures", 
			fontweight='demi', fontsize=15,
			pad=13)

		cell_phase_formatting = {
						'meanG1': (40, 'center', 'bottom', (0, 0.01)),
						'S': (85, 'left', 'top', (0.02, -0.01)),
						'G2/M': (100, 'center', 'top', (0, -0.06))}
		plot_example_cluster(cluster_num=11, x_metric_index=0, y_metric_index=3,
				xlim=(-0.3, 0.3), ylim=(-0.65, 0.65), 
				cell_phase_formatting=cell_phase_formatting, ax=ax1)
		ax1.set_title("Temporally offset\nmeasures (CW)", 
			fontweight='demi', fontsize=15,
			pad=13)

		cell_phase_formatting = {
						'meanG1': (40, 'right', 'top', (-0.0125, -0.01)),
						'S': (85, 'left', 'center', (0.065, 0.0)),
						'G2/M': (110, 'left', 'bottom', (0.01, 0.01))}
		plot_example_cluster(gene_name='CLB1', x_metric_index=2, y_metric_index=3,
				xlim=(-2, 2),
				ylim=(-2, 2), cell_phase_formatting=cell_phase_formatting, ax=ax2)
		ax2.set_title("Temporally offset\nmeasures (CCW)", 
			fontweight='demi', fontsize=15, pad=13)
		plt.subplots_adjust(hspace=0.4)

		save_figure_for_paper(f"{self.save_dir}/example_annotated_trajectory.png")

		
	def create_joined_ptrs_dataset(self, fig_metrics):
		chromatin_ptrs = fig_metrics.chromatin_processor.deconvolved_chromatin_ptrs

		joined_metrics_ptr_data = None
		for key in chromatin_ptrs.keys():
			current_df = chromatin_ptrs[key].loc[self.metrics_gene_data.gene_row_index]
			if joined_metrics_ptr_data is None:
				joined_metrics_ptr_data = current_df
			else:
				joined_metrics_ptr_data = joined_metrics_ptr_data.join(current_df)

		expression_ptrs = fig_metrics.expression_processor.all_transcripts_ptrs.loc[\
			self.metrics_gene_data.gene_row_index][['ptr']]

		joined_metrics_ptr_data = joined_metrics_ptr_data.join(expression_ptrs)
		columns = list(joined_metrics_ptr_data.columns[:-1]) + ['expression_ptr']

		joined_metrics_ptr_data.columns = columns
		return joined_metrics_ptr_data


	def initialize_tf_processor(self):

		from src.TranscriptionFactorProcessor import TranscriptionFactorProcessor

		# todo: parent object reference is getting a bit confusing
		fig_metrics = self.fig_metrics

		tf_processor = TranscriptionFactorProcessor(fig_metrics.output_dir,
			expression_processor=fig_metrics.integration.expression_processor,
			chromatin_processor=fig_metrics.integration.chromatin_processor)
		tf_processor.build_comprehensive_dataframe()
		self.tf_processor = tf_processor

	def perform_tf_analysis_of_clusters(self):

		tf_processor = self.tf_processor
		tf_genes_df = tf_processor.comprehensive_w_associated_genes_df
		tf_genes_df = tf_genes_df[~tf_genes_df.associated_gene.isna()][['associated_gene']].reset_index().set_index(
			['associated_gene', 'tf'])

		# Next, run through each cluster and examine which TFs are associated with each cluster
		# Evaluate, how we consider a cluster is enriched for a TF.
		orf_cluster_assignments = pd.DataFrame(self.clusters, index=self.filtered_set_orfnames,
					columns=['cluster'])

		# Pivot the dataframe
		binary_tf_table = tf_genes_df.pivot_table(
			index='associated_gene',
			columns='tf',
			aggfunc='size',
			fill_value=0
		)

		# Convert counts to binary (True/False or 1/0)
		binary_tf_table = (binary_tf_table > 0).astype(int)

		cluster_tf_indicator_df = orf_cluster_assignments.join(binary_tf_table, how='inner')
		cluster_tf_indicator_df = cluster_tf_indicator_df.reset_index()\
			.sort_values('cluster')

		# Group by cluster and sum the TF indicators
		cluster_tf_summary = cluster_tf_indicator_df.groupby('cluster')[
			['Abf1', 'Ace2', 'Bas1', 'Cha4', 'Cin5', 'Crz1', 'Cup9', 'Fhl1', 
			 'Ste12', 'Stp2', 'Stp4', 'Sum1', 'Swi4', 'Tbf1', 'Ume6', 'Urc2', 'Yap1', 'Yrr1']
		].sum()

		# Or more simply, exclude the 'index' column if it exists:
		tf_columns = [col for col in cluster_tf_indicator_df.columns if col not in ['index', 'cluster']]
		cluster_tf_summary = cluster_tf_indicator_df.groupby('cluster')[tf_columns].sum()

		print("Only include TFs with at 2 or more total sites in the set.")

		self.filtered_cluster_tf_summary = cluster_tf_summary.loc[:, (cluster_tf_summary > 1).any(axis=0)]
		self.cluster_tf_summary = cluster_tf_summary

		print(f"Filtering from {cluster_tf_summary.sum().sum()} sites to {self.filtered_cluster_tf_summary.sum().sum()}.")

	def layout_chromatin_transcription_figure(self):
		"""
		Layout main chromatin-transcription figure.

		1. Global relationship
		2. Chromatin Clustering
		3. Cluster examples

		"""
		# Create compositor

		from pipeline.figure_composer import FigureCompositor
		compositor = FigureCompositor(1024, 650, debug_mode=True)

		self.figures_dir = self.save_dir
		chrom_metrics_figs_dir = self.fig_metrics.figures_dir
		
		# Define image paths for the three chromatin metrics plots
		image_paths = [
			f'{chrom_metrics_figs_dir}/expression_chromatin_deconvolved_scatter.png',
			f'{self.figures_dir}/clusters_heatmap.png',
			f'{self.save_dir}/example_annotated_trajectory.png',
			f'{self.figures_dir}/trajectories/meta_cluster_11.png',
			f'{self.figures_dir}/trajectories/meta_cluster_13.png',
		]

		from pipeline.figure_composer_helpers import layout_images_horizontally,\
			place_image_below, add_panel_labels_to_images,\
			layout_images_vertically
		
		# Layout images vertically with equal proportions
		placed_images = layout_images_horizontally(
			compositor,
			[image_paths[0], image_paths[2]],
			width_proportions=[0.71, 0.29],
			between_padding=80,
			margin=(10, 10),
			available_width=640,
			image_keys=['ptrs', 'diagram']
		)

		hm_width = placed_images['ptrs']['logical_size'][0]+80
		place_image_below(compositor, image_paths[1], 'ptrs',
			width=hm_width,
			new_key='heatmap')

		placed_images = layout_images_vertically(
			compositor,
			[image_paths[3], image_paths[4]],
			widths=[250, 250],
			x_position=750,
			between_padding=30,
			margin=(10, 10),
			image_keys=['clust11', 'clust13']
		)

		# Add panel labels (abcd)
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			labels="acbfg",
			font_size=30,
			offsets=[
				(0, 20),
				(0, 20),
				(0, 20),
				(0, 20),
				(0, 20),
			]
		)

		compositor.add_panel_label_to_image('diagram', 'd', offset=(0, 226), font_size=30)

		compositor.add_panel_label_to_image('diagram', 'e', offset=(0, 430), font_size=30)
		
		# Save the composed figure
		output_path = f'{self.fig_metrics.panel_figures_dir}/Figure5_Chromatin_Transcription.png'
		compositor.save(output_path)
		
		print(f"Panel layout saved to: {output_path}")
		
		return output_path
