
import pandas as pd
from cc_src.read_bam import _fromRoman


def extract_desc_val(data, key):
	"""Extract the values in the description field of SGD"""
	def _extract_desc_val_row(orf_row, key):
		"""Parse the description of a orf_row in sgd gff, extract `keys` and return 
		as a series. Can be used to return as a df when called with apply"""

		des_map = {}
		for entry in orf_row.desc.split(';'):
			k, val = tuple(entry.split('='))
			if k == key:
				return val

		return None

	# parse description column to extract relevant columns
	vals = data.apply(lambda row: _extract_desc_val_row(row, key=key),
		axis=1)

	return vals


def read_sgd_genes(filename='data/reference_data/sgd_R64-1-1_20110208.gff'):
	"""Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

	data = pd.read_csv(filename, sep='\t', skiprows=19, 
							  names=["chr", "source", "cat", "start", "stop", ".", 
								  "strand", "", "desc"])
	data = data[data.columns[[0, 2, 3, 4, 6, 8]]]
	data.columns = ["chr", "cat", "start", "stop", "strand", "desc"]
	data = data[data['cat'] == 'gene']

	gene_names = extract_desc_val(data, 'gene') 
	systematic_name = extract_desc_val(data, 'ID') 
	orf_classification = extract_desc_val(data, 'orf_classification') 
		
	data['gene'] = gene_names
	data['orf_name'] = systematic_name
	data['classification'] = orf_classification

	data = data[['orf_name', 'gene', "chr", "cat", "start", "stop", "strand", 
		'classification', ]]
	data['length'] = data['stop'] - data['start']

	return data.set_index('orf_name')

def get_gene(genename_or_orfname):
	genes = read_sgd_genes()
	found_genes = genes[(genes['gene'] == genename_or_orfname) | (genes.index == genename_or_orfname)]

	if len(found_genes) == 0:
		raise ValueError(f"Could not find gene {genename_or_orfname}")

	return found_genes.iloc[0]

def get_gene_name(orf_name):
	return get_gene(orf_name).gene


def get_orfname(gene_name):
	gene = get_gene(gene_name)
	return gene.name


def get_gene_name_orf_name(genename_or_orfname):

	generow = get_gene(genename_or_orfname)

	return generow.name, generow.gene


def get_gene_coordinates(gene_name):
	gene = get_gene(gene_name)
	return gene.start, gene.stop, gene.strand, gene.chr

def read_park_TSS_PAS():

	TSS_filename = 'data/reference_data/Park_2014_TSS_V64.gff'
	PAS_filename = 'data/reference_data/Park_2014_PAS_V64.gff'

	def _read_park_gff(filename, key):
		park_data = pd.read_csv(filename, sep='\t', skiprows=3)
		columns = park_data.columns[[0, 2, 3]] # relevant columns
		park_data = park_data[columns].copy()

		# cleanup
		park_data.columns = ['chr', 'orf_name', key]
		park_data.orf_name = park_data.orf_name.str.replace('_%s' % key, '')
		park_data[key] = park_data[key].astype(int)

		return park_data.set_index('orf_name')

	TSS = _read_park_gff(TSS_filename, 'TSS')

	# manually annotated/adjusted TSSs for vignettes
	TSS.loc['YBR072W', 'TSS'] = 381753
	TSS.loc['YDR253C', 'TSS'] = 964767
	TSS.loc['YBR294W', 'TSS'] = 789000
	TSS.loc['YLR092W', 'TSS'] = 323500
	TSS.loc['YOL164W', 'TSS'] = 6000

	PAS = _read_park_gff(PAS_filename, 'PAS')
	data = TSS[['TSS']].join(PAS[['PAS']])

	data['manually_curated'] = False
	data.loc['YBR072W', 'manually_curated'] = True
	data.loc['YDR253C', 'manually_curated'] = True
	data.loc['YBR294W', 'manually_curated'] = True
	data.loc['YLR092W', 'manually_curated'] = True
	data.loc['YOL164W', 'manually_curated'] = True

	return data


def read_sgd_chromosomes(filename='data/reference_data/sgd_R64-1-1_20110208.gff'):
	"""Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

	data = pd.read_csv(filename, sep='\t', skiprows=19, 
							  names=["chr", "source", "cat", "start", "stop", ".", 
								  "strand", "", "desc"])
	data = data[data.columns[[0, 2, 3, 4, 6, 8]]]
	data.columns = ["chr", "cat", "start", "stop", "strand", "desc"]
	data = data[data['cat'] == 'chromosome']
	chroms = data.chr.str.replace('chr', '').apply(_fromRoman)
	
	data['chr'] = chroms
	data = data.set_index('chr')[['stop']].rename(columns={'stop': 'length'})
	
	return data

def get_chromosome_length(chrom):
	return read_sgd_chromosomes().loc[chrom]['length']


def read_nondubious_genes_dataset():
	genes_nondub = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')
	return genes_nondub
