
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class TFBindingSites:
	"""Class to find and plot TF binding sites, from Kelliher, Rossi, MacIsaac, (eventually if needed, 
	FIMO)"""

	def __init__(self):

		from glob import glob
		from io import StringIO

		bed_filenames = glob('data/reference_data/rossi_chip_exo/04_ChExMix_Peaks/*bed')
		filename = bed_filenames[0]

		def read_rossi_bed(filename):
		    with open(filename, 'r') as f:
		        lines = f.read()
		        tf = filename.split('/')[-1].split('_')[0]
		        df = pd.read_csv(StringIO(lines), sep='\t', header=None)
		        df.columns = ['chr', 'start', 'stop', 'identifier', 'value', '.']
		        df['tf'] = tf
		    return df

		# Load the list of cell cycle TFs from Kelliher, 2018
		cctf_df = pd.read_csv('data/reference_data/mbc-29-2644-s002.csv', skiprows=2,
		           header=0)
		keep_rows = ~cctf_df['Common Name'].isna()
		cctf_df = cctf_df.loc[keep_rows].set_index('Standard Name')[['Common Name', 'Cell-Cycle Function']]

		all_rossi_tf_dfs = pd.DataFrame()
		for filename in bed_filenames:
		    tf_df = read_rossi_bed(filename)
		    all_rossi_tf_dfs = pd.concat([all_rossi_tf_dfs, tf_df])
		chroms = all_rossi_tf_dfs.chr.str.replace('chr', '').astype(int)
		all_rossi_tf_dfs.chr = chroms

		macisaac_tfs = pd.read_csv('data/reference_data/p005_c2.sacCer3.gff.txt',
		    sep='\t', header=None)
		macisaac_tfs = macisaac_tfs[8].str.replace('Site ',
		    '').str.replace(';', '').unique()
		print(f"There are {len(macisaac_tfs)} MacIsaac TFs")

		# Intersect the set of cell cycle transcription factors from Kelliher
		# with the list of ChIP-exo binding sites. 
		cctfs = cctf_df['Common Name'].values
		rossi_tfs = all_rossi_tf_dfs.tf.str.upper().unique()

		print(f"There are {len(cctfs)} cell cycle transcription factors (CCTFs) profiled by Kelliher, 2018")
		print(f"There are {len(rossi_tfs)} transcription factors profiled using ChIP-exo by Rossi, 2021")

		kelliher_rossi_intersection = np.array(list(set(cctfs).intersection(set(rossi_tfs))))
		print(f"There are {len(kelliher_rossi_intersection)} CCTFS in the Rossi dataset.")

		kelliher_macisaac_intersection = set(macisaac_tfs).intersection(set(cctfs))
		print(f"There are {len(kelliher_macisaac_intersection)} CCTFS in the MacIsaac dataset.")

		self.cctf_df = cctf_df
		self.all_rossi_tf_dfs = all_rossi_tf_dfs
		self.kelliher_rossi_intersection = kelliher_rossi_intersection
		self.cctfs = cctfs

		self.cctf_colors = {}
		for i in range(len(self.kelliher_rossi_intersection)):
			tf = self.kelliher_rossi_intersection[i].upper()
			self.cctf_colors[tf] = plt.get_cmap('tab10')(i)


	def filter_tf_binding_sites(self, gene, mnase_span):
		"""Filter for TF binding sites at a gene location"""

		def filter_rossi(chr, search_span):
			filtered_sites = self.all_rossi_tf_dfs[(self.all_rossi_tf_dfs.chr == chr) & 
											  (self.all_rossi_tf_dfs.start > search_span[0]) & 
											  (self.all_rossi_tf_dfs.start < search_span[1])]
			return filtered_sites

		filtered_rossi_sites = filter_rossi(gene.chr, mnase_span)
		filtered_cc_tfs = filtered_rossi_sites[filtered_rossi_sites.tf.str.upper().isin(self.cctfs)]

		self.filtered_cc_tfs = filtered_cc_tfs


	def plot_tf_sites(self, ax, legend=False):

		from matplotlib.lines import Line2D
		import matplotlib.patches as mpatches
		handles, labels = plt.gca().get_legend_handles_labels()

		tf_mid = ((self.filtered_cc_tfs.start+self.filtered_cc_tfs.stop)//2).values

		tfs_w_handles = set()

		for i in range(len(tf_mid)):
			tf = self.filtered_cc_tfs.iloc[i].tf

			x, y = tf_mid[i], 15
			if i % 2 == 0: y += 15
			color = self.cctf_colors[tf.upper()]
			ax.scatter(x, y, edgecolor=color, marker='v', s=9, facecolor='none', label=tf)

			if tf not in tfs_w_handles:
				# Create handles for legend
				tf_handle = Line2D([0], [0], lw=0, 
					label=f'{tf}', markeredgecolor=color, markeredgewidth=1,
					markerfacecolor='none', markersize=3, marker='v')
				handles.extend([tf_handle])
				tfs_w_handles = tfs_w_handles.union({tf})

		if legend:
			plt.legend(handles=handles, bbox_to_anchor=(1.2, 2.5), loc='upper left')

	def select_tf(self, selected_tf):

		rossi_sites = self.all_rossi_tf_dfs
		tf_sites = rossi_sites[rossi_sites.tf == selected_tf]

		tf_sites = tf_sites.sort_values(['chr', 'start']).reset_index(drop=True)

		# First check if a site is near its neigbor
		def filter_neighbors(tf_sites, window=200):
		    """To avoid overcounting, filter out tf sites that are near neighbors.
		    Currently we will default to keeping the first of the two instances.
		    
		    todo: May need to come up with a better plan for which of the two sites to keep.
		    """

		    # Recursively run the process until no sites are left to drop 
		    original_sites = tf_sites.copy()
		    total_dropped = []
		    while True:
		        drop_sites = []

		        for i in range(0, len(tf_sites)-1):

		            first = tf_sites.iloc[i]
		            second = tf_sites.iloc[i+1]

		            if (first.chr == second.chr) and (second.start-first.start < window):
		                drop_sites.append(second.name)
		        
		        tf_sites = tf_sites.drop(drop_sites)
		        total_dropped = total_dropped + drop_sites

		        if len(drop_sites) == 0: break
		    return tf_sites.reset_index(drop=True), original_sites.loc[total_dropped]

		window = 200
		filtered_tf_sites, dropped_sites = filter_neighbors(tf_sites, window)

		print(f"Filtered out {len(dropped_sites)} sites which were within {window} of a nearby site")

		self.filtered_tf_sites = filtered_tf_sites
