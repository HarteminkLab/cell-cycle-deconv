

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
