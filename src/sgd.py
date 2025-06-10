import numpy as np
import pandas as pd
from src.read_bam import _fromRoman


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


def load_aux_annotations():
    """Read non-genic, non-origin, annotations of interest. Useful for plotting."""
    
    # Select subset of categories to include, from: sgd_all_annotations.cat.unique()
    categories_of_interest = ['centromere', 'repeat_region', 'tRNA',
                              'rRNA', 'long_terminal_repeat']

    sgd_all_annotations = read_sgd_file(convert_chr_from_roman=True)
    name = extract_desc_val(sgd_all_annotations, 'Name')
    sgd_all_annotations['name'] = name

    aux_annotations = sgd_all_annotations[sgd_all_annotations.cat.isin(categories_of_interest)
                                         & (sgd_all_annotations.chr > 0)].copy()
    return aux_annotations


def read_sgd_file(filename='data/reference_data/sgd_R64-1-1_20110208.gff',
	convert_chr_from_roman=False):
	"""Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

	data = pd.read_csv(filename, sep='\t', skiprows=19, 
							  names=["chr", "source", "cat", "start", "stop", ".", 
								  "strand", "", "desc"])
	data = data[data.columns[[0, 2, 3, 4, 6, 8]]]
	data.columns = ["chr", "cat", "start", "stop", "strand", "desc"]

	if convert_chr_from_roman:
		data.chr = data.chr.str.replace('chr', '').apply(_fromRoman)

	return data

def read_sgd_genes(filename='data/reference_data/sgd_R64-1-1_20110208.gff',
	remove_chr_roman=False):
	"""Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

	data = read_sgd_file(filename)
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

	if remove_chr_roman:
		data['chr'] = data.chr.str.replace('chr', '').apply(_fromRoman)

	return data.set_index('orf_name')


def get_gene(genename_or_orfname, genes=read_sgd_genes()):
	found_genes = genes[(genes['gene'] == genename_or_orfname) | (genes.index == genename_or_orfname)]

	if len(found_genes) == 0:
		raise ValueError(f"Could not find gene {genename_or_orfname}")

	return found_genes.iloc[0]

def get_gene_name(orf_name, genes=read_sgd_genes()):
	return get_gene(orf_name, genes).gene


def get_gene_title_name(gene_or_orf_name, genes=read_sgd_genes(), include_system=True):
	"""For displaying gene names, avoid displaying None"""

	orf_name, gene_name = get_gene_name_orf_name(gene_or_orf_name)

	if gene_name is None:
		gene_title = ("$\\it{" + orf_name + "}$")
	else:

		if include_system:
			gene_title = ("$\\it{" + gene_name + "}$ / $\\it{" + orf_name + "}$")
		else:
			gene_title = ("$\\it{" + gene_name + "}$")

	return gene_title


def get_orfnames(gene_names):
	return [get_orfname(g) for g in gene_names]
	

def get_genenames(orfnames):
	return [get_gene_name_orf_name(o)[1] for o in orfnames]


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


def load_origins_sgd():
    sgd = read_sgd_file()
    origins_sgd = sgd[sgd['cat'] == 'ARS'].copy()
    arsname = extract_desc_val(origins_sgd, 'Name')
    origins_sgd['ars_name'] = arsname
    chroms = origins_sgd.chr.str.replace('chr', '').apply(_fromRoman)
    origins_sgd.chr = chroms
    return origins_sgd


def read_sgd_w_go():

	from pandas import Series

	data = read_sgd_file()
	orfs = data[data['cat'] == 'gene'].copy()
	orfs['orf_name'] = None
	orfs['orf_class'] = None

	# chromosomal orfs
	chroms = orfs.chr.str.replace('chr', '').apply(_fromRoman)
	orfs.chr = chroms
	orfs = orfs[orfs.chr > 0]

	orf_names = extract_desc_val(orfs, 'ID') 
	orf_classes = extract_desc_val(orfs, 'orf_classification')
	names = extract_desc_val(orfs, 'gene') 
	ontology = extract_desc_val(orfs, 'Ontology_term') 

	orfs['orf_name'] = orf_names
	orfs['orf_class'] = orf_classes

	# set name if it exists, or orf_name by default
	orfs['name'] = names
	orfs.loc[orfs.name.isna(), 'name'] = orfs[orfs.name.isna()]['orf_name']
	orfs['ontology'] = ontology

	orfs['length'] = orfs['stop'] - orfs['start'] + 1
	orfs = orfs.set_index('orf_name')[[
		'name','chr','start','stop','length','strand','orf_class', 'ontology']]

	return orfs


def get_intergenic_regions(genes_df, buffer_genes=500, buffer_chrom_end=10000,
		min_threshold_len=500):
	"""
	Find intergenic regions meeting buffer criteria from gene annotations.
	
	Args:
		genes_df: DataFrame with columns [orf_name, TSS, PAS]
		buffer_genes: Buffer distance from genes (bp)
		buffer_chrom_end: Buffer from chromosome ends (bp)
	
	Returns:
		DataFrame with columns [chromosome, start, end]
	"""
	# Extract chromosome from orf_name and add as column
	genes_df = genes_df.copy()
	
	intergenic_regions = []
	
	# Process each chromosome
	for chrom in range(1, 17):
		# Get chromosome length
		chrom_length = get_chromosome_length(chrom)
		
		# Get genes for this chromosome and sort by position
		chrom_genes = genes_df[genes_df['chr'] == chrom].copy()
		
		# Sort genes by leftmost position (min of TSS/PAS)
		chrom_genes['left_pos'] = chrom_genes[['TSS', 'PAS']].min(axis=1)
		chrom_genes = chrom_genes.sort_values('left_pos')
		
		# Find gaps between genes
		for i in range(len(chrom_genes) - 1):
			current_gene_end = max(chrom_genes.iloc[i]['TSS'], 
								 chrom_genes.iloc[i]['PAS'])
			next_gene_start = min(chrom_genes.iloc[i+1]['TSS'],
								chrom_genes.iloc[i+1]['PAS'])
			
			# Calculate potential intergenic region
			region_start = current_gene_end + buffer_genes
			region_end = next_gene_start - buffer_genes
			
			# Check if region is valid
			if region_end > region_start:
				intergenic_regions.append({
					'chromosome': chrom,
					'start': region_start,
					'end': region_end
				})
		
		# Check chromosome ends
		# Left end
		first_gene_start = min(chrom_genes.iloc[0]['TSS'], 
							 chrom_genes.iloc[0]['PAS'])
		if first_gene_start > (buffer_chrom_end + buffer_genes):
			intergenic_regions.append({
				'chromosome': chrom,
				'start': buffer_chrom_end,
				'end': first_gene_start - buffer_genes
			})
			
		# Right end
		last_gene_end = max(chrom_genes.iloc[-1]['TSS'],
						   chrom_genes.iloc[-1]['PAS'])
		if (chrom_length - last_gene_end) > (buffer_chrom_end + buffer_genes):
			intergenic_regions.append({
				'chromosome': chrom,
				'start': last_gene_end + buffer_genes,
				'end': chrom_length - buffer_chrom_end
			})
	intergenic_regions_df = pd.DataFrame(intergenic_regions)
	intergenic_regions_df['length'] = intergenic_regions_df.end - intergenic_regions_df.start

	intergenic_regions_df = intergenic_regions_df[intergenic_regions_df['length'] >= min_threshold_len]

	return intergenic_regions_df


def select_genes_in_window(orfs, chrom, span, orf_classes=
	['Verified', 'Uncharacterized', 'Dubious']):

	if 'left_end' in orfs.columns and 'right_end' in orfs.columns:
		genes = orfs[(orfs['chr'] == chrom) & 
		 			 ((orfs['right_end'] >= span[0]) & 
		   			  (orfs['left_end'] <= span[1])) &
		 			  (orfs.classification.isin(orf_classes))]
	else:
		genes = orfs[(orfs['chr'] == chrom) & 
		 			 ((orfs['stop'] >= span[0]) & 
		   			  (orfs['start'] <= span[1])) &
		 			  (orfs.classification.isin(orf_classes))]

	return genes


def construct_orf_annotation_dataset():
	"""Create dataset for orf annotation using park TSS and PAS. Define the left and right
	end boundaries for easy window finding computation"""
	from src.sgd import read_park_TSS_PAS, read_nondubious_genes_dataset

	genes = read_nondubious_genes_dataset()
	gene_TSS_PASs = read_park_TSS_PAS()

	joined_genes_TSS_PASs = genes[genes.columns[~genes.columns.isin(['TSS', 'PAS'])]].join(
	gene_TSS_PASs[['TSS', 'PAS']])


	joined_genes_TSS_PASs = joined_genes_TSS_PASs.rename(columns={'TSS': 'park_TSS', 'PAS': 'park_PAS'})

	joined_genes_TSS_PASs['TSS'] = joined_genes_TSS_PASs['park_TSS']
	joined_genes_TSS_PASs['PAS'] = joined_genes_TSS_PASs['park_PAS']

	is_watson = genes['strand'] == '+'
	tss_is_nan = np.isnan(joined_genes_TSS_PASs.TSS)
	pas_is_nan = np.isnan(joined_genes_TSS_PASs.PAS)

	def set_field_w_existing_col(dat, selection, key, val_key):
	    dat.loc[selection, key] = dat.loc[selection, val_key]
	    return dat

	set_field_w_existing_col(joined_genes_TSS_PASs, (is_watson & tss_is_nan), 'TSS', 'start')
	set_field_w_existing_col(joined_genes_TSS_PASs, (is_watson & pas_is_nan), 'PAS', 'stop')
	set_field_w_existing_col(joined_genes_TSS_PASs, (~is_watson & tss_is_nan), 'TSS', 'stop')
	set_field_w_existing_col(joined_genes_TSS_PASs, (~is_watson & pas_is_nan), 'PAS', 'start')

	set_field_w_existing_col(joined_genes_TSS_PASs, (is_watson & tss_is_nan), 'TSS', 'start')
	set_field_w_existing_col(joined_genes_TSS_PASs, (is_watson & pas_is_nan), 'PAS', 'stop')
	set_field_w_existing_col(joined_genes_TSS_PASs, (~is_watson & tss_is_nan), 'TSS', 'stop')
	set_field_w_existing_col(joined_genes_TSS_PASs, (~is_watson & pas_is_nan), 'PAS', 'start')


	set_field_w_existing_col(joined_genes_TSS_PASs, (is_watson), 'left_end', 'TSS')
	set_field_w_existing_col(joined_genes_TSS_PASs, (is_watson), 'right_end', 'PAS')
	set_field_w_existing_col(joined_genes_TSS_PASs, (~is_watson), 'left_end', 'PAS')
	set_field_w_existing_col(joined_genes_TSS_PASs, (~is_watson), 'right_end', 'TSS')

	return joined_genes_TSS_PASs
