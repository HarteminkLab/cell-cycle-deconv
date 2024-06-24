
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
