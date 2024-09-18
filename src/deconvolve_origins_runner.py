
import sys
sys.path.append('.')

import pandas as pd
from src.utils import print_fl, mkdirs_safe

from src.timer import Timer
from src.model import Model
from matplotlib import pyplot as plt
from src.combined_chromatin_model import CombinedChromatinModel


def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument

	Usage:

		<output> <gamma_value/-1> <config_type: shared/distinct/delta> <origin_index>

	Will run the find optimal gamma procedure on the chromatin


	Script for shared G1 config model with a fixed gamma value and copy number correction

	python src/deconvolve_origins_runner.py output/deconvolve_sharedg1_0066_cc 0.0066 shared 10
	python src/deconvolve_origins_runner.py output/deconvolve_distinctg1_0066_cc 0.0066 distinct 10

	"""

	system_args = tuple(sys.argv)

	print_fl(f"System arguments:\t{system_args}")

	# Specify output directory, gamma value and gene index
	(_, out_dir, gamma, config_type, origin_index) = system_args
	origin_index = int(origin_index)
	gamma = float(gamma)

	print("Config type: ", config_type)

	# Find optimal gamma
	if gamma < 0: gamma = None

	sys.stdout.flush()

	from src.origins import load_origins_w_replication

	origins = load_origins_w_replication(full=True)
	origin = origins.iloc[origin_index]

	print_fl(f"Index: [{origin_index}/{len(origins)}] Deconvolving combined model, origin: {origin['ars_name']}/{origin.name}...")

	if gamma is None:
		print_fl(f"No gamma specified, finding optimal gamma value for chromatin.")

	print_fl(f"Output to: {out_dir}")

	# -------------------------

	plot_dir = f'{out_dir}/plots'
	chromatin_out_dir = f'{out_dir}/chromatin'
	mkdirs_safe([chromatin_out_dir, plot_dir])

	# ----------------------

	# Load the configuration for the combined chromatin
	from src.config import load_configs_by_config_type

	config1, config2 = load_configs_by_config_type(config_type, with_copy_correction=False)

	# ----------------------

	combined_model = CombinedChromatinModel(config1, config2)
	combined_model.load_combined_mnase_orc(origin.name)
	combined_model.setup_deconv_model()

	if gamma is not None:
		combined_model.deconvolve(verbose=False, gamma=gamma)
	else:
		combined_model.deconvolve_find_optimal_gamma()

	# Find and track +1 and -1 nucleosome position
	combined_model.find_origin_p1_and_m1_nucleosome_position()

	# -------------- Save the output ------------------

	combined_model.save_origin_plots(plot_dir, origin_index)
	combined_model.save_deconvolved_origin_outputs(chromatin_out_dir, origin_index)


if __name__ == '__main__':
	main()
