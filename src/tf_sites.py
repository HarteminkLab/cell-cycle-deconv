
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class TFBindingSites:
	"""Class to find and plot TF binding sites, from Kelliher, Rossi, MacIsaac, (eventually if needed, 
	FIMO)"""

	def __init__(self):

		from src.reference_data import read_all_rossi_sites
		from src.reference_data import read_kelliher_cc_tfs

		# Load rossi sites
		all_rossi_tf_dfs = read_all_rossi_sites()

		# Load the list of cell cycle TFs from Kelliher, 2018
		kelliher_cctfs = read_kelliher_cc_tfs()

		# Intersect the set of cell cycle transcription factors from Kelliher
		# with the list of ChIP-exo binding sites. 
		kelliher_cctfs_names = kelliher_cctfs['Common Name'].values

		rossi_tfs = all_rossi_tf_dfs.tf.str.upper().unique()

		#print(f"There are {len(kelliher_cctfs)} cell cycle transcription factors (CCTFs) profiled by Kelliher, 2018")
		#print(f"There are {len(rossi_tfs)} transcription factors profiled using ChIP-exo by Rossi, 2021")

		kelliher_rossi_intersection = np.array(list(set(kelliher_cctfs_names).intersection(set(rossi_tfs))))
		# print(f"There are {len(kelliher_rossi_intersection)} CCTFS in the Rossi dataset.")

		self.all_rossi_tf_dfs = all_rossi_tf_dfs
		self.kelliher_rossi_intersection = kelliher_rossi_intersection
		self.all_kelliher_cell_cycle_tfs = kelliher_cctfs

		from src.TranscriptionFactorProcessor import read_transcription_factor_set_from_go

		rossi_tfs = self.all_rossi_tf_dfs.tf.unique()
		kelliher_cell_cycle_tfs = self.kelliher_rossi_intersection.copy()

		all_rossi_tfs = [tf.upper() for tf in rossi_tfs]
		go_binding_genes, go_terms = read_transcription_factor_set_from_go()
		go_rossi_tfs = set(all_rossi_tfs).intersection(set(go_binding_genes))
		cell_cycle_rossi_tfs = set(list(go_rossi_tfs)).intersection(kelliher_cell_cycle_tfs)

		print(f"Number of total factors profiled by Rossi: ", len(all_rossi_tfs))
		print(f"Number of TFs selected by GO terms ({','.join(go_terms)}): ", len(go_binding_genes))
		print(f"Number of TFs with Rossi binding sites: ", len(go_rossi_tfs))
		print(f"   of these, Kelliher cell-cycle TFS: ", len(cell_cycle_rossi_tfs))

		# Subset the rossi binding set for the TFs (go selected)
		# This is the final set of sites we will be analyzing
		filtered_tf_names = [tf.title() for tf in go_rossi_tfs]
		self.filtered_rossi_go_tf_binding_sites = self.all_rossi_tf_dfs.set_index('tf').loc[filtered_tf_names].reset_index()

		self.go_terms_for_tfs = go_terms
		self.tfs_with_tf_go_term = go_binding_genes
		self.rossi_tfs_with_tf_go = go_rossi_tfs
		self.cell_cycle_rossi_tfs = cell_cycle_rossi_tfs
