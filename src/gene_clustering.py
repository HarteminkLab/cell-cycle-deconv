
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import correlate
from sklearn.cluster import DBSCAN
from src.timer import Timer
from sklearn_extra.cluster import KMedoids
from src.figure_configs import FiguresConfig
from src.global_config import GlobalConstants
from src.deconv_data import load_f_files
from src.config import load_configs_by_config_type
from src.geneset import get_deconvolved_geneset
from src.deconvolved_f_plotter import plot_pseudo_gene


class GeneClustering:
	"""Compute the gene clusters of expression"""

	def __init__(self, outdir):

		from src.Figure3_Chrom_Gene_Expression import Figure3_Chrom_GeneExpression
		figures = Figure3_Chrom_GeneExpression(outdir)
		gene_expression = figures.gene_expression_a.gene_expression_f
		geneset = get_deconvolved_geneset()

		self.gene_expression = gene_expression
		self.geneset = geneset

		config1, config2 = load_configs_by_config_type('shared')
		self.config = config1


	def set_max_expression_threshold(self, q_cutoff = 0.5):
		# Filter out genes with low maximal expression, to remove genes that cycle with
		# low expression.
		
		t_indices = self.config.get_Hpositions_for_branch('t')
		gene_expression_top_branch = self.gene_expression[t_indices]

		quantile_expression = gene_expression_top_branch.quantile(q_cutoff, axis=1)
		q_val = np.quantile(quantile_expression, q_cutoff)
		quantile_q_genes = quantile_expression[quantile_expression > q_val].index
		print(f"Genes with maximal expression for quantile: {q_cutoff}\nis set at {q_val:.2f} expression level. {len(quantile_q_genes)} genes meet this criteria")

		self.high_max_expression_genes = quantile_q_genes


	def set_threshold_ptr(self, q_ptr):

		t_indices = self.config.get_Hpositions_for_branch('t')
		gene_expression_top_branch = self.gene_expression[t_indices].loc[self.high_max_expression_genes]

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
		plt.hist(top_gene_expression_ptr_df, bins=50)
		plt.yscale('log')
		plt.xlim(0.9, 4)
		plt.axvline(q_val, c='red')
		plt.title("Gene expression, PTR")

		self.q_val = q_val
		self.q_ptr = q_ptr
		self.top_gene_expression_ptr_df = top_gene_expression_ptr_df
		self.high_ptr_gene_expression = high_ptr_gene_expression

	def set_gene_expression_to_cluster(self, expression, normalize=True):

		if normalize:
			data_to_cluster = (expression - \
				expression.mean(axis=1).values.reshape((-1, 1))) /\
				expression.std(axis=1).values.reshape((-1, 1))
			data_to_cluster = data_to_cluster
		else:
			data_to_cluster = expression

		self.data_to_cluster_df = data_to_cluster
		self.data_to_cluster = data_to_cluster.values

	def compute_distance_from_gene_expression(self):
		timer = Timer()

		# Previous clustering data
		# high_ptr_gene_expression = self.high_ptr_gene_expression
		# normalized_high_ptr_expression = (high_ptr_gene_expression - \
		# 	high_ptr_gene_expression.mean(axis=1).values.reshape((-1, 1))) /\
		# 	high_ptr_gene_expression.std(axis=1).values.reshape((-1, 1))
		# time_series_data = normalized_high_ptr_expression.values

		time_series_data = self.data_to_cluster

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


	def cluster_expression(self, num_clusters):
		self.kmedoids, labels, medoid_indices = cluster_timeseries_kmedoids(self.distance_matrix, 
			num_clusters)
		labels = labels+1
		clustered_expression = self.data_to_cluster_df
		clustered_expression['cluster'] = labels
		self.clustered_expression = clustered_expression.reset_index().set_index(['cluster', 'orf_name'])

		self.labels = labels
		self.medoid_indices = medoid_indices
		self.num_clusters = num_clusters

	def compute_medoids(self):
		medoids = np.zeros((len(self.medoid_indices), \
			self.data_to_cluster.shape[1]))
		for i, medoid_index in enumerate(self.medoid_indices):
			medoids[i] = self.data_to_cluster.iloc[medoid_index]
		self.medoids = medoids

	def load_chromatin_for_cluster(self, cluster):

		current_cluster_exp = self.clustered_expression.loc[cluster]
		orf_names = current_cluster_exp.index.values

		def load_chromatin_and_shift(chromatin_dir, orf_names):

			# Load chromatin from disk and shift
			accumulated_gene_f_imgs = load_chromatin_from_dir(chromatin_dir, orf_names)

			# Shift the gene f images for their designated cluster
			shifted_gene_f_imgs = self.shift_cluster_chromatin_data(accumulated_gene_f_imgs)

			return accumulated_gene_f_imgs, shifted_gene_f_imgs

		if not self.selected_cluster == cluster:

			self.selected_cluster = cluster
			self.cluster_TSS_f_imgs, self.shifted_TSS_f_imgs = load_chromatin_and_shift(
				self.TSS_chromatin_dir, orf_names)
			self.cluster_PAS_f_imgs, self.shifted_PAS_f_imgs = load_chromatin_and_shift(
				self.PAS_chromatin_dir, orf_names)

		else:
			print(f"Already loaded chromtin for cluster {cluster}, reverting to cache.")


	def plot_clustered_heatmap(self):

		from src.chromatin_model import draw_phase_label_annotations

		plt.figure(figsize=(4, 4))

		cluster_counts = self.clustered_expression.sort_index().reset_index()\
			.groupby('cluster').count()[['orf_name']].rename(columns={'orf_name': 'count'})
		cluster_counts['cumulative_sum'] = cluster_counts.cumsum()['count']
		ylabel_midpoints = (cluster_counts['count'] // 2).values + np.concatenate([[0], 
			cluster_counts['cumulative_sum'].values[:-1]])
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

		config = self.config
		draw_phase_label_annotations(plt.gca(), config, flip=True, 
			annotations_x=len(clustered_data)+28)
		plt.ylim(cluster_counts['cumulative_sum'].values[-1]+50, 0)


	def shift_cluster_chromatin_data(self, current_cluster_chromatin):
		"""Shift the chromatin data such that the assigned cluster for the gene is optimally aligned
		to the medoid of the cluster"""

		cluster = self.selected_cluster
		current_shifts = self.aligned_shifts_df[self.aligned_shifts_df.cluster == cluster]

		def shift_chrom_data(chrom_dat, current_shifts):
			shifted_chrom_dat = chrom_dat.copy()
			for i in range(shifted_chrom_dat.shape[0]):
				shift = int(current_shifts.iloc[i]['shift'])
				shifted_chrom_dat[i] = np.roll(chrom_dat[i], shift, axis=0)
			return shifted_chrom_dat

		aligned_cluster_chromatin_data = shift_chrom_data(current_cluster_chromatin, current_shifts)
		return aligned_cluster_chromatin_data


	def plot_chromatin_in_cluster(self):

		aligned_TSS_imgs = self.shifted_TSS_f_imgs.mean(axis=0)
		aligned_PAS_imgs = self.shifted_PAS_f_imgs.mean(axis=0)
		plot_h_positions = get_phase_indices_for_plotting()
		title = f"Cluster {self.selected_cluster}, n={len(self.shifted_TSS_f_imgs)}"
		fig = plot_chromatin(aligned_TSS_imgs, aligned_PAS_imgs, plot_h_positions, title=title)
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

		fig = plt.figure(figsize=(5, 3))

		plot_marker_xs = get_phase_indices_for_plotting()
		aligned_expression_df = self.aligned_expression_df.reset_index().set_index(['cluster', 'orf_name'])
		medoid = self.medoids[cluster-1]
		curves = aligned_expression_df.loc[cluster]

		x = self.config.get_timepoints_for_branch('t')

		plt.plot(x, curves.T, c='#ddd', alpha=1.)
		plt.plot(x, medoid)
		plt.ylim(-4, 4.1)
		plt.xlim(x[0], x[-1])
		plt.xticks([])
		plt.yticks([])
		plt.ylabel("Normalized expression")

		cluster_medoid_df = pd.DataFrame({
			'time': x,
			'H_pos': self.clustered_expression.columns,
			'medoid_value': medoid
		})
		cluster_medoid_df = cluster_medoid_df.set_index('H_pos')

		for i in range(len(plot_marker_xs)):
			marker_x = plot_marker_xs[i]
			tp = cluster_medoid_df.loc[marker_x].time
			y_val = cluster_medoid_df.loc[marker_x].medoid_value
			plt.scatter(tp, y_val, facecolor='#555', edgecolor='#555',
				marker='D', s=23, zorder=100, lw=1)
			plt.text(tp, y_val+0.25, str(i+1), va='bottom', ha='center')

		plt.title(f"Cluster {cluster}, n={len(curves)}",
			fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)

		from src.chromatin_model import draw_phase_label_annotations
		config = self.config
		draw_phase_label_annotations(plt.gca(), config, flip=True, annotations_x=-3.7)


	def set_chromatin_dirs(self, TSS_chromatin_dir, PAS_chromatin_dir):
		self.TSS_chromatin_dir = TSS_chromatin_dir
		self.PAS_chromatin_dir = PAS_chromatin_dir


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
	"""Compute the circular correlation between two vectors and return the 
	max correlation and the optimal shift."""
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





def load_chromatin_from_dir(chromatin_dir, orf_names):

	config, _ = load_configs_by_config_type('shared')
	t_indices = config.get_Hpositions_for_branch('t')

	all_gene_f_df = load_f_files(chromatin_dir, orf_names)

	# Flip the strand of the loaded chromatin
	geneset = get_deconvolved_geneset()
	current_genes = geneset[['gene', 'strand']].loc[orf_names]
	current_genes['gene_index'] = np.arange(len(current_genes))

	accumulated_gene_f_imgs = all_gene_f_df.values.reshape(
		(-1, 178, *GlobalConstants.IMAGE_SHAPE)).astype(float)[:, t_indices]
	crick_genes = current_genes[current_genes.strand == '-'].gene_index
	accumulated_gene_f_imgs[crick_genes] = np.flip(
		accumulated_gene_f_imgs[crick_genes], axis=3)

	return accumulated_gene_f_imgs


def plot_chromatin(aligned_TSS_imgs, aligned_PAS_imgs, plot_h_positions=None, vmax=5,
	title=None):

	# Plot predefined h positions
	if plot_h_positions is None:
		plot_h_positions = get_phase_indices_for_plotting()

	config, _ = load_configs_by_config_type('shared')
	t_h_indices = config.get_Hpositions_for_branch('t')
	t_indices = np.arange(len(t_h_indices))
	plot_t_indices = [t_indices[t_h_indices == h][0] for h in plot_h_positions]
	indices_df = pd.DataFrame({'value': plot_t_indices})

	k = len(indices_df)

	fig, axs = plt.subplots(3, 3, figsize=(21, 5))
	annotation_axs = axs[0] # Axs for pseudo-gene annotations
	axs = np.array(axs[1:]).T.flatten()


	for i in range(len(annotation_axs)):

		ax = annotation_axs[i]
		ax.set_xticks([])
		ax.set_yticks([])

		plot_psuedo_gene(ax)

		if i == 0: ax.set_title('Shared G1', fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		elif i == 1: ax.set_title('S', fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
		elif i == 2: ax.set_title('G2/M', fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)

	for i, index_row in indices_df.iterrows():

		ax = axs[i]

		tss_img = aligned_TSS_imgs[index_row.value].astype(float)
		pas_img = aligned_PAS_imgs[index_row.value].astype(float)

		ax.set_ylabel(f"{i+1}", fontsize=FiguresConfig.FIG_LABEL_FONTSIZE, 
			rotation=0, ha='right', labelpad=9)

		ax.imshow(tss_img,
				  origin='lower', 
				  cmap='magma_r', aspect='auto', 
				  vmax=vmax,
				  extent=GlobalConstants.BIN_EXTENTS)

		ax.axvline(0, c='#333', alpha=0.5, lw=1, ls='dotted')
		ax.axvline(GlobalConstants.GB_LEN*2, c='#333', alpha=0.5, lw=1, ls='dotted')

		pas_extents = [GlobalConstants.BIN_EXTENTS[0]+GlobalConstants.GB_LEN*2,
			GlobalConstants.BIN_EXTENTS[1]+GlobalConstants.GB_LEN*2,
			GlobalConstants.BIN_EXTENTS[2], GlobalConstants.BIN_EXTENTS[3]]
		ax.imshow(pas_img,
				  origin='lower', 
				  cmap='magma_r', aspect='auto', 
				  vmax=vmax,
				  extent=pas_extents)

		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlim(-GlobalConstants.PROM_LEN, GlobalConstants.GB_LEN*3)
		ax.axvline(GlobalConstants.GB_LEN, c='#aaa', ls='solid', lw=1)

		if i == 1:
			pas_img_center = GlobalConstants.GB_LEN*2
			ax.set_xticks([-300, 0, 300, GlobalConstants.GB_LEN, 
				pas_img_center-300, pas_img_center, pas_img_center+300])
			ax.set_xticklabels(['-300', '+1 nuc.', '+300', '', 
								'-300', 'last nuc.', '+300'])

	plt.subplots_adjust(top=0.82)
	plt.suptitle(title, fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)

	return fig



def get_phase_indices_for_plotting():

	from src.helpers import proportion_indices

	config, _ = load_configs_by_config_type('shared')
	g1_indices = config.get_Hpositions_for_phase('CG1')
	s_indices = config.get_Hpositions_for_phase('S')
	g2m_indices = config.get_Hpositions_for_phase('G2M')

	sel_g1_indices = proportion_indices(g1_indices, [0.2, 0.8])
	sel_s_indices = proportion_indices(s_indices, [0.2, 0.8])
	sel_g2m_indices = proportion_indices(g2m_indices, [0.2, 0.8])

	plot_marker_xs = np.concatenate([
		sel_g1_indices,
		sel_s_indices,
		sel_g2m_indices
	])
	return plot_marker_xs

