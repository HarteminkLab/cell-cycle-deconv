

# Currently depecrated, may bring this back
# 3/5/25



# import glob
# import math

# import numpy as np
# import pandas as pd

# from src.plot_helpers import color_for_key
# from matplotlib import pyplot as plt
# from src.chromatin_model import read_chromosome_mnase_reads
# from src.reference_data import load_spellman_orfs, load_analysis_genes
# from src.config import load_yl_rg1_vst_config
# from src.deconv_data import load_gene_expression_fs


# class GeneExpressionAnalysis:
# 	"""Analysis to classify and characterize the deconvolved gene expression"""

# 	def __init__(self, gene_expression_dir):
# 		self.gene_expression_dir = gene_expression_dir
# 		self.gene_expression_f = load_gene_expression_fs(gene_expression_dir)

# 	def compute_ptrs(self):

# 		from src.config import load_configs_by_config_type

# 		config1, config2 = load_configs_by_config_type('shared')
# 		t_indices = config1.get_Hpositions_for_branch('t')

# 		from src.peak_to_trough import compute_quantile_ptr

# 		gene_ptrs_df = self.gene_expression_f[[]].copy()
			
# 		for orf_name, row in self.gene_expression_f.iterrows():
# 			ptr = compute_quantile_ptr(row[t_indices])
# 			gene_ptrs_df.loc[orf_name, 'ptr'] = ptr

# 		self.gene_ptrs_df = gene_ptrs_df
