
import pandas as pd


def load_origins(full=False):
	"""Load the origins of replications as defined by Belsky, 2015.

	Option for full 798 origins set or short 239 origins with G1 and G2 footprints"""
	from src.read_bam import _fromRoman
	origins = pd.read_csv('data/reference_data/oridb_acs_feature_file_239origins_sacCer3_102920_belsky2015.csv').set_index('name')

	if full:
		origins = pd.read_csv('data/reference_data/oridb_acs_feature_file_jab-curated-798-sites_sacCer3_belsky2015.csv').set_index('name')

	origins.chr = origins.chr.str.replace('chr', '').apply(_fromRoman)
	return origins


# Deprecated, we need a way to easily load genomic items with the correponding replication timing.
#
#
# def load_origins_w_replication(full=False):

# 	from src.config import load_default_chrom_configs

# 	# Load origins of replication
# 	# load the replication timing
# 	# assign the replication index and timing to each origin for analysis and plotting
# 	origins = load_origins(full=full)
# 	from src.stepwise_replication_solver import replication_timing_from_index

# 	# Depiction of the chromosome 10 replication timing profile computed from the MNase-seq
# 	chrom_replication_profile = pd.read_csv(
# 		'data/replication_timing/yl_2019/chrom_replication_timing_shared.csv')
# 	chrom_replication_profile = chrom_replication_profile.set_index(['chr', 'start'])

# 	# Copy number correction procedure....
# 	config1, config2 = load_default_chrom_configs

# 	origins_repl = origins.copy()
		
# 	for chrom in range(1, 17):

# 		repl_idx = chrom_replication_profile.loc[chrom].replication_index
# 		repl_timing = replication_timing_from_index(repl_idx, config1, config2)

# 		origins_chr = origins[origins.chr == chrom]

# 		from src.global_config import GlobalConstants
# 		from src.sgd import get_chromosome_length
# 		from src.mnase_replication_timing_analysis import get_bin_for_position

# 		for origin_name, origin in origins_chr.iterrows():

# 			bin_idx, bin_start_pos = get_bin_for_position(origin.pos, repl_idx.index)
# 			replication_index = repl_idx.iloc[bin_idx]
# 			replication_time = repl_timing[bin_idx]

# 			origins_repl.loc[origin_name, 'replication_time'] = replication_time
# 			origins_repl.loc[origin_name, 'replication_index'] = replication_index

# 	return origins_repl



def get_origin_title_name(origin):
	"""For displaying gene names, avoid displaying None"""

	title = ("$\\it{" + origin.ars_name + "}$")

	return title


def identify_origins_of_interest(origin_dataset, genes_dataset, origins_of_interest,
	gene_names_of_interest, window=6000):
	"""
	Identify origins that have genes of interest within a specified window.
	"""

	from src.sgd import get_orfnames
	import pandas as pd
	
	# Filter genes to those of interest that exist in the dataset
	orfnames = get_orfnames(gene_names_of_interest)
	genes_filtered = genes_dataset.loc[orfnames].copy()
	
	# Initialize result dataframe with False values
	result = pd.DataFrame(
		index=origin_dataset.index, 
		columns=orfnames,
		dtype=bool
	)
	result[:] = False
	
	# For each origin
	for origin_idx in origin_dataset.index:
		origin_row = origin_dataset.loc[origin_idx]
		origin_chr = origin_row['chr']
		origin_pos = origin_row['pos']
		
		# Define window around origin (window/2 on each side)
		half_window = window // 2
		window_start = origin_pos - half_window
		window_end = origin_pos + half_window
		
		# Check each gene of interest on the same chromosome
		same_chr_genes = genes_filtered[genes_filtered['chr'] == origin_chr]
			
		for gene_name in same_chr_genes.index:
			gene_row = same_chr_genes.loc[gene_name]
			gene_start = gene_row['start']
			gene_end = gene_row['stop']

			# Check if gene overlaps with window around origin
			if gene_start <= window_end and gene_end >= window_start:
				result.loc[origin_idx, gene_name] = True
	
	return result

def identify_interesting_origins():

	from src.origins import load_origins
	from src.sgd import read_nondubious_genes_dataset

	genes = read_nondubious_genes_dataset()

	eff_key = 'derived_origin_efficiency_from_mcguffee_et_al_2013'
	origins = load_origins(full=True).sort_values(eff_key, ascending=False)
	interesting_origins = origins[(origins['activation_time'] == 'early')
	        & ~(origins.mcm_loading_class.isna()) & 
	       (origins[eff_key] > 0.5)]
	interesting_origins

	# Define key gene sets based on literature
	MCM_GENES = ['MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7']
	REPLICATION_GENES = ['CDC6', 'CDC45', 'DBF4', 'ORC1', 'ORC2', 'ORC3', 'ORC4', 'ORC5', 'ORC6']
	CYCLIN_GENES = ['CLN1', 'CLN2', 'CLN3', 'CLB1', 'CLB2', 'CLB3', 'CLB4', 'CLB5', 'CLB6']

	# Combined gene set
	ALL_CELL_CYCLE_GENES = MCM_GENES + REPLICATION_GENES + CYCLIN_GENES
	gene_names_of_interest = ALL_CELL_CYCLE_GENES

	cc_origins_genes_intersection = identify_origins_of_interest(interesting_origins, genes, 
		interesting_origins, 
		gene_names_of_interest, window=12000)

