
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
