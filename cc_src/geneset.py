

def cyclin_genes():

	genes = [
		# three G1 cyclins:
		'CLN1', 'CLN2', 'CLN3',

		# Two S-phase cyclins:
		'CLB5', 'CLB6',

		# And four mitotic cyclins:
		'CLB1', 'CLB2', 'CLB3', 'CLB4']

	return genes


def positive_control_genes():

	genes = [

		# B-type cyclins
		"CLB2", "CLN2",

		# RNR complex
		"RNR1", "RNR3",

		# Alpha factor genes
		"FIG1", "FIG2",

		"RAD51", "RAD53",
		
		# Daughter-specific expression genes
		"DSE1", "DSE2", "DSE3", "DSE4"
	]

	return genes

def get_sorted_geneset():
	"""Get the list of genes to deconvolve first. Sorted by custom priority"""

	import pandas as pd

	geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')

	#starting_set = set(cyclin_genes()).union(set(positive_control_genes()))

	# Get spellman genes
	#gene_ast = pd.read_csv('datasets/datasets_from_web_deconvolution.cs.duke.edu/gene_associated.tsv', sep='\t')
	#spellman_orfs = gene_ast[gene_ast.Spellman1998 == '1']['Systematic Name'].values

	#sorted_geneset = geneset.copy()
	#sorted_geneset['priority'] = 1000

	# Spellman genes second
	#sorted_geneset.loc[spellman_orfs, 'priority'] = 2

	# Selected set of genes first
	#sorted_geneset.loc[geneset['gene'].isin(starting_set), 'priority'] = 1

	# Sorted set of genes
	#sorted_geneset = sorted_geneset.sort_values('priority')

    # todo: back to unsorted, for debugging failed genes 
    # may not be necessary to sort, as the runs don't take too long at this point
	return geneset
