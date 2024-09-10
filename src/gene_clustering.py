import numpy as np
from scipy.signal import correlate
from sklearn.cluster import DBSCAN
from src.timer import Timer
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import pandas as pd
from src.figure_configs import FiguresConfig


class GeneClustering:
	"""Compute the gene clusters of expression"""

	def __init__(self, outdir):

		from src.Figure3_Chrom_Gene_Expression import Figure3_Chrom_GeneExpression
		figures = Figure3_Chrom_GeneExpression(outdir)
		gene_expression = figures.gene_expression_a.gene_expression_f

		from src.geneset import get_deconvolved_geneset
		geneset = get_deconvolved_geneset()

		self.gene_expression = gene_expression
		self.geneset = geneset


	def set_threshold_ptr(self, q_ptr):

		from src.config import load_configs_by_config_type

		config1, config2 = load_configs_by_config_type('shared')
		t_indices = config1.get_Hpositions_for_branch('t')
		gene_expression_top_branch = self.gene_expression[t_indices]

		from src.peak_to_trough import compute_quantile_ptr_2d

		top_gene_expression_ptr = compute_quantile_ptr_2d(gene_expression_top_branch, hi=0.9, lo=0.1)
		top_gene_expression_ptr_df = gene_expression_top_branch[[]].copy()
		top_gene_expression_ptr_df['ptr'] = top_gene_expression_ptr

		# Select a set of genes in the upper 99% percentile
		q_val = np.quantile(top_gene_expression_ptr_df, q_ptr)
		quantile_ptr_genes = top_gene_expression_ptr_df[top_gene_expression_ptr_df.ptr > q_val]
		print(f"Selected {len(quantile_ptr_genes)} genes in the top 80% with quantile value greater"
			  f" than {q_val:.2f}")

		high_ptr_gene_expression = gene_expression_top_branch.loc[quantile_ptr_genes.index]

		plt.figure(figsize=(4, 3))
		plt.hist(top_gene_expression_ptr_df, bins=100)
		plt.yscale('log')
		plt.xlim(0.9, 4)
		plt.axvline(q_val, c='red')
		plt.title("Gene expression, PTR")

		self.config = config1
		self.q_val = q_val
		self.q_ptr = q_ptr
		self.top_gene_expression_ptr_df = top_gene_expression_ptr_df
		self.high_ptr_gene_expression = high_ptr_gene_expression


	def compute_distance_from_gene_expression(self):
		timer = Timer()

		high_ptr_gene_expression = self.high_ptr_gene_expression
		normalized_high_ptr_expression = (high_ptr_gene_expression - \
			high_ptr_gene_expression.mean(axis=1).values.reshape((-1, 1))) /\
			high_ptr_gene_expression.std(axis=1).values.reshape((-1, 1))

		time_series_data = normalized_high_ptr_expression.values

		# Compute the pairwise distance matrix using circular correlation
		n_samples = len(time_series_data)
		distance_matrix = np.zeros((n_samples, n_samples))

		for i in range(n_samples):
			for j in range(i + 1, n_samples):
				distance = circular_distance(time_series_data[i], time_series_data[j])
				distance_matrix[i, j] = distance
				distance_matrix[j, i] = distance
				
			if i % 100 == 0:
				timer.print_time(f"{i}/{n_samples}")

		timer.print_time()

		self.distance_matrix = distance_matrix
		self.normalized_high_ptr_expression = normalized_high_ptr_expression


	def cluster_expression(self, num_clusters):
		self.kmedoids, labels, medoid_indices = cluster_timeseries_kmedoids(self.distance_matrix, num_clusters)
		labels = labels+1
		clustered_expression = self.high_ptr_gene_expression.copy()
		clustered_expression['cluster'] = labels
		self.clustered_expression = clustered_expression.reset_index().set_index(['cluster', 'orf_name'])
		self.labels = labels
		self.medoid_indices = medoid_indices
		self.num_clusters = num_clusters

	def compute_medoids(self):
		medoids = np.zeros((len(self.medoid_indices), \
			self.normalized_high_ptr_expression.shape[1]))
		for i, medoid_index in enumerate(self.medoid_indices):
			medoids[i] = self.normalized_high_ptr_expression.iloc[medoid_index]
		self.medoids = medoids

	def load_chromatin_for_cluster(self, chromatin_dir, cluster):
		from src.deconv_data import load_f_files

		t_indices = self.config.get_Hpositions_for_branch('t')

		current_cluster_exp = self.clustered_expression.loc[cluster]
		all_gene_f_df = load_f_files(chromatin_dir, current_cluster_exp)
		self.current_cluster_chromatin = all_gene_f_df
		self.current_cluster_exp = current_cluster_exp
		self.selected_cluster = cluster

		# Flip the strand of the loaded chromatin
		current_genes = self.geneset[['gene', 'strand']].loc[self.current_cluster_exp.index]
		current_genes['gene_index'] = np.arange(len(current_genes))

		accumulated_gene_f_imgs = self.current_cluster_chromatin.values.reshape(
			(-1, 178, 23, 91)).astype(float)[:, t_indices]
		crick_genes = current_genes[current_genes.strand == '-'].gene_index
		accumulated_gene_f_imgs[crick_genes] = np.flip(accumulated_gene_f_imgs[crick_genes],\
			axis=3)

		self.current_cluster_chromatin = accumulated_gene_f_imgs
		self.normalize_chromatin()

	def normalize_chromatin(self):
		# Normalize chrom data by total sum per each image
		chrom_data = self.current_cluster_chromatin
		normalized_chrom_data = chrom_data.copy()
		total_target = 1000

		for gene_i in range(chrom_data.shape[0]):
			gene_chrom = chrom_data[gene_i]
			gene_sum_per_t = gene_chrom.sum(axis=1).sum(axis=1)
			normalized_gene_chrom = gene_chrom / gene_sum_per_t.reshape((-1, 1, 1))
			normalized_chrom_data[gene_i] = normalized_gene_chrom * total_target

		self.normalized_chrom = normalized_chrom_data

	def plot_clustered_heatmap(self):

		from src.chromatin_model import draw_phase_label_annotations

		plt.figure(figsize=(4, 4))

		cluster_counts = self.clustered_expression.sort_index().reset_index()\
			.groupby('cluster').count()[['orf_name']].rename(columns={'orf_name': 'count'})
		cluster_counts['cumulative_sum'] = cluster_counts.cumsum()['count']
		ylabel_midpoints = (cluster_counts['count'] // 2).values + np.concatenate([[0], cluster_counts['cumulative_sum'].values[:-1]])
		cluster_counts['ylabels'] = ylabel_midpoints

		plt.title("Clustered\nGene Expression", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		yticks = cluster_counts.ylabels
		ytick_labels = cluster_counts.index
		plt.yticks(yticks, ytick_labels)
		plt.xticks([])

		for cluster, row in cluster_counts.iterrows():
			plt.axhline(row.cumulative_sum, c='black', lw=1)
			
		plt.gca().yaxis.set_tick_params(pad=3, length=0)
		plt.ylabel("Cluster")

		clustered_data = self.aligned_expression_df.reset_index()\
			.set_index(['cluster', 'orf_name']).sort_index().astype(float)
		clustered_data.columns = self.clustered_expression.columns

		config = self.config
		cg1_indices = config.get_Hpositions_for_phase("CG1")
		postg1_indices = config.get_Hpositions_for_phase("postG1")
		cg1_tps = config.get_phase_timepoints_for_phase("CG1")
		postg1_tps = config.get_phase_timepoints_for_phase("postG1")

		cg1_tx = clustered_data[cg1_indices]
		pg1_tx = clustered_data[postg1_indices]
		n = len(pg1_tx)

		cg1_extent = [cg1_tps[0], cg1_tps[-1], 0, n]
		postg1_extent = [cg1_tps[-1], postg1_tps[-1], 0, n]

		plt.imshow(cg1_tx, cmap='Purples',
			interpolation='none', aspect='auto', origin='lower', extent=cg1_extent, vmin=-3, vmax=3)
		plt.imshow(pg1_tx, cmap='Purples',
			interpolation='none', aspect='auto', origin='lower', extent=postg1_extent, vmin=-3, vmax=3)
		plt.colorbar()

		config = self.config
		draw_phase_label_annotations(plt.gca(), config, flip=True, 
			annotations_x=len(clustered_data)+28)
		plt.ylim(cluster_counts['cumulative_sum'].values[-1]+50, 0)


	def shift_cluster_chromatin_data(self):

		cluster = self.selected_cluster
		current_shifts = self.aligned_shifts_df[self.aligned_shifts_df.cluster == cluster]

		def shift_chrom_data(chrom_dat, current_shifts):
			shifted_chrom_dat = chrom_dat.copy()
			for i in range(shifted_chrom_dat.shape[0]):
				shift = int(current_shifts.iloc[i]['shift'])
				shifted_chrom_dat[i] = np.roll(chrom_dat[i], shift, axis=0)
			return shifted_chrom_dat

		self.aligned_cluster_chromatin_data = shift_chrom_data(self.current_cluster_chromatin, current_shifts)
		self.aligned_normalized_cluster_chromatin_data = shift_chrom_data(self.normalized_chrom, current_shifts)

		self.current_cluster_chromatin_mean = self.current_cluster_chromatin.mean(axis=0)
		self.current_aligned_chromatin_mean = self.aligned_cluster_chromatin_data.mean(axis=0)

		self.current_normalized_chromatin_mean = self.normalized_chrom.mean(axis=0)
		self.current_normalized_aligned_chromatin_mean = self.aligned_normalized_cluster_chromatin_data.mean(axis=0)


	def plot_chromatin_in_cluster(self, vmax=5, full=False):

		cluster = self.selected_cluster
		
		indices_df = self.interesting_points_df[self.interesting_points_df.cluster == cluster].sort_values('value')
		indices_df = indices_df.reset_index(drop=True)
	
		column_titles = ['Unaligned', 'Normalized Unaligned', 'Aligned', 'Normalized Aligned']

		# Normalized chromatin data
		normalized_chrom_dat = self.current_normalized_chromatin_mean
		normalized_aligned_chromatin = self.current_normalized_aligned_chromatin_mean

		# Unnormalized chromatin data
		chrom_dat = self.current_cluster_chromatin_mean
		aligned_chromatin = self.current_aligned_chromatin_mean
		k = len(indices_df)

		if full:
			cols = 3
			fig = plt.figure(figsize=(16, k*1.5))
		else:
			cols = 1
			fig = plt.figure(figsize=(5, k*1.5))

		cluster_index = cluster-1
		medoid = self.medoids[cluster_index]

		for i, index_row in indices_df.iterrows():
			plt.subplot(k, cols, (i*cols)+1)

			# Unaligned data
			unaligned = chrom_dat[index_row.value].astype(float)
			aligned = aligned_chromatin[index_row.value].astype(float)

			if full:
				plt.imshow(unaligned,
						  origin='lower', cmap='magma_r', aspect='auto', vmax=vmax)
				plt.xticks([])
				plt.yticks([])
				plt.ylabel(f"{index_row.point_type}: {index_row.value}")

				if i == 0: plt.title("Unaligned")

				plt.subplot(k, cols, (i*cols)+2)
				plt.imshow(aligned,
						  origin='lower', cmap='magma_r', aspect='auto', vmax=vmax)
				plt.xticks([])
				plt.yticks([])

				if i == 0: plt.title("Aligned")

				plt.subplot(k, cols, (i*cols)+3)
				plt.imshow(aligned-unaligned,
						  origin='lower', cmap='RdBu_r', aspect='auto', vmin=-0.05, vmax=0.05)
				plt.xticks([])
				plt.yticks([])
				if i == 0: plt.title("Aligned-Unaligned")
				plt.subplots_adjust(top=0.87)
			else:
				plt.ylabel(f"{index_row.point_type}: {index_row.value}")
				plt.subplot(k, cols, (i*cols)+1)
				plt.imshow(aligned,
						  origin='lower', cmap='magma_r', aspect='auto', vmax=vmax)
				plt.xticks([])
				plt.yticks([])
				plt.subplots_adjust(top=0.9)

		plt.suptitle(f"Cluster {cluster}, n={len(self.aligned_cluster_chromatin_data)}",
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)

		return fig

	def compute_alignments_and_interesting_points_per_cluster(self):

		from src.gene_expression_processing import retrieve_important_points
			
		expression = self.normalized_high_ptr_expression
		medoids = self.medoids
		num_clusters = self.num_clusters

		interesting_points_df = pd.DataFrame()
		aligned_shifts_df = pd.DataFrame()
		aligned_expression_df = pd.DataFrame()

		for i in range(num_clusters):

			medoid = medoids[i]
			cluster = i+1
			cluster_expression = expression[self.labels == cluster]
			aligned_expression, aligned_shift = align_to_medoid(cluster_expression, medoid)
				
			smoothed_medoid, important_points, _ = retrieve_important_points(medoid, 
																			min_distance=20)
			important_points['cluster'] = cluster
			aligned_shift['cluster'] = cluster
			aligned_expression['cluster'] = cluster

			interesting_points_df = pd.concat([interesting_points_df, important_points])
			aligned_shifts_df = pd.concat([aligned_shifts_df, aligned_shift])    
			aligned_expression_df = pd.concat([aligned_expression_df, aligned_expression])

		self.aligned_expression_df = aligned_expression_df
		self.interesting_points_df = interesting_points_df
		self.aligned_shifts_df = aligned_shifts_df


	def plot_aligned_cluster(self, cluster):

		aligned_expression_df = self.aligned_expression_df.reset_index().set_index(['cluster', 'orf_name'])
		medoid = self.medoids[cluster-1]
		curves = aligned_expression_df.loc[cluster]

		x = self.config.get_timepoints_for_branch('t')

		plt.plot(x, curves.T, c='#ddd', alpha=1.)
		plt.plot(x, medoid)
		plt.ylim(-4, 4.1)
		plt.xticks([])
		plt.yticks([])

		cluster_medoid_df = pd.DataFrame({
			'time': x,
			'H_pos': aligned_expression_df.columns,
			'medoid_value': medoid
		})

		important_points = self.interesting_points_df[self.interesting_points_df.cluster == cluster]

		for index, row in important_points.iterrows():
			if row.point_type == 'maxima':
				color = 'red'
				marker='^'
			elif row.point_type =='minima':
				color = 'blue'
				marker='v'
			else:
				color = 'black'
				marker='o'

			hpos = aligned_expression_df.columns[int(row.value)]
			tp = cluster_medoid_df.loc[hpos].time
			y_val = cluster_medoid_df.loc[hpos].medoid_value

			plt.scatter(tp, y_val, facecolor='none', edgecolor=color, 
				marker=marker, s=42, zorder=100, lw=1)

		plt.title(f"Cluster {cluster}, n={len(curves)}",
			fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)

		from src.chromatin_model import draw_phase_label_annotations
		config = self.config
		draw_phase_label_annotations(plt.gca(), config, flip=True, annotations_x=-3.7)


	def plot_aligned_clusters(self):
		from src.gene_expression_processing import retrieve_important_points
			
		expression = self.normalized_high_ptr_expression
		medoids = self.medoids

		plt.figure(figsize=(8, 23))

		num_clusters = self.num_clusters
		columns = 2

		interesting_points_df = self.interesting_points_df
		aligned_shifts_df = self.aligned_shifts_df
		aligned_expression_df = self.aligned_expression_df.reset_index().set_index(['cluster', 'orf_name'])

		for i in range(num_clusters):

			cluster = i+1
			medoid = medoids[i]
			cluster_expression = expression[self.labels == cluster]

			expression_curves = cluster_expression.T
			aligned_expression_curves = aligned_expression_df.loc[cluster].T
			
			plt.subplot(num_clusters, 2, (columns*i)+1)
			plt.plot(expression_curves, c='#ddd', alpha=1.0)
			plt.plot(cluster_expression.columns, medoid)
			plt.ylabel(f"Cluster {cluster}, n={len(cluster_expression)}", rotation=0,
				ha='right', fontsize=13, labelpad=10)
			plt.xticks([])
			# plt.yticks([])
			if i == 0: plt.title("Unaligned")
			plt.ylim(-4, 4)

			plt.subplot(num_clusters, 2, (columns*i)+2)
			plt.plot(cluster_expression.columns, aligned_expression_curves, c='#ddd', alpha=1.0)
			plt.plot(cluster_expression.columns, medoid)
			plt.xticks([])
			plt.yticks([])
			plt.ylim(-4, 4)
			if i == 0: plt.title("Aligned")

			important_points = interesting_points_df[interesting_points_df.cluster == cluster]


			for index, row in important_points.iterrows():
				if row.point_type == 'maxima':
					color = 'red'
					ls='solid'
				elif row.point_type =='minima':
					color = 'blue'
					ls='solid'
				else:
					color = 'black'
					ls = 'dotted'

				plt.axvline(cluster_expression.columns[int(row.value)], color=color,
					ls=ls)

		plt.subplots_adjust(hspace=0.125, wspace=0.125, top=0.9)


	def run_gene_ontology_on_clustered_genes(self, clustered_genes, gene_ontology=None):
		"""clustered genes is expected to be indexed on cluster number and orf names"""

		from src.gene_ontology import GeneOntology
		gene_ontology = GeneOntology()

		self.go_results, self.go_results_sig = run_gene_ontology_on_clustered_genes(
			self.geneset, clustered_genes, gene_ontology)


	def print_go_results(self):
		sig_res = self.go_results[['id', 'name', 'fdr_bh', 'cluster', 'study_items']]
		sig_res = sig_res[sig_res.fdr_bh < 0.01]

		go_clusters = sig_res.cluster.unique()
		for clust in go_clusters:
			print(f"Cluster {clust}")
			go_names = (sig_res[sig_res.cluster == clust]['name'].values)
			for go_name in go_names:
				print("  " + go_name[0].upper() + go_name[1:])
			print()


def circular_correlation(v1, v2, return_idx=False):
	"""Compute the circular correlation between two vectors and return the max correlation and the optimal shift."""
	# Ensure the vectors are numpy arrays
	v1 = np.array(v1)
	v2 = np.array(v2)
	
	# Get the length of the vectors
	n = len(v1)
	
	# Compute the cross-correlation
	corr = correlate(v1, v2, mode='full')
	
	# Normalize the correlation
	norm_factor = np.linalg.norm(v1) * np.linalg.norm(v2)
	normalized_corr = corr / norm_factor
	
	# Only consider the relevant part of the cross-correlation (circular part)
	circular_corr = np.concatenate([normalized_corr[-(n-1):], normalized_corr[:n]])
	
	# Find the maximum correlation and its corresponding shift
	max_corr = np.max(circular_corr)
	optimal_shift = np.argmax(circular_corr[:n])

	# Since shifts beyond n-1 actually mean shifts in the opposite direction
	if optimal_shift >= n:
		optimal_shift = optimal_shift - n
	
	if return_idx:
		return max_corr, optimal_shift

	return max_corr


def circular_distance(series1, series2):
	"""Convert circular correlation to a distance measure."""
	return 1 - circular_correlation(series1, series2)


def cluster_timeseries_kmedoids(dist_matrix, n_clusters):
	"""
	Perform K-medoids clustering using the precomputed distance matrix.
	"""
	kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', 
						method='alternate', init='k-medoids++', random_state=123)
	kmedoids.fit(dist_matrix)
	return kmedoids, kmedoids.labels_, kmedoids.medoid_indices_


def align_to_medoid(data, medoid):
	aligned_data = []

	data_shift = data[[]].copy()
	data_shift[['shift']] = None
	for idx, series in data.iterrows():  
		corr, shift = circular_correlation(medoid, 
										series, return_idx=True)
		aligned_data.append(np.roll(series, shift))
		data_shift.loc[idx, 'shift'] = shift

	return pd.DataFrame(aligned_data, index=data.index), data_shift


def run_gene_ontology_on_clustered_genes(geneset, clustered_genes, gene_ontology=None):
	"""clustered genes is expected to be indexed on cluster number and orf names"""

	from src.gene_ontology import GeneOntology

	if gene_ontology is None:
		gene_ontology = GeneOntology()
		
	def get_results_go(gene_ontology, k):

		all_results = pd.DataFrame()
		all_sigfig = pd.DataFrame()
		gene_names = geneset[['gene']]

		for i in range(k):
			cluster = i + 1

			# Get the current cluster orfs and join with gene names
			selected_orfs = clustered_genes.loc[cluster].index.values
			selected_gene_names = gene_names.loc[selected_orfs]['gene'].values

			# Run GO on the gene names
			gene_ontology.run_go(selected_gene_names)

			results = gene_ontology.results_df.copy()
			results_sigfig = gene_ontology.results_sig_df.copy()
			
			results['cluster'] = cluster
			results_sigfig['cluster'] = cluster

			all_results = pd.concat([all_results, results])
			all_sigfig =  pd.concat([all_sigfig, results_sigfig])

		all_results = all_results[~all_results['name'].isin(['biological_process', 
			'molecular_function', 'cellular_component'])]
		all_sigfig = all_sigfig[~all_sigfig['name'].isin(['biological_process', 
			'molecular_function', 'cellular_component'])]
		
		return all_results, all_sigfig

	num_clusters = clustered_genes.reset_index().cluster.max()
	return get_results_go(gene_ontology, num_clusters)
