
import pandas as pd


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


def get_missing_geneset():
	"""Missing genes from the 2/20/24 runs, temporary function to deconvolve the missing 1000 genes"""
	geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies_missing_temp_2024-02-27.csv').set_index('orf_name')
	return geneset


def get_sorted_geneset():
	"""Get the list of genes to deconvolve first. Sorted by custom priority"""

	geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')

	# TODO: Deconvolving missing 1000 genes

	return get_missing_geneset()
