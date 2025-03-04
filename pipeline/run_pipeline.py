
import sys
sys.path.append('.')

import sys
from src.utils import mkdirs_safe, parse_bool, print_fl


def main():

	system_args = tuple(sys.argv)
	command = system_args[1]

	# 1. Deconvolve individual replication profiles, learn cell cycle parameters from MNase-seq
	if command == 'replication':

		(_, command, output_directory, replicate, chrom, num_epochs, cold_start) = system_args

		mkdirs_safe([output_directory])

		cold_start = parse_bool(cold_start)
		chrom = int(chrom)
		replicate = int(replicate)
		num_epochs = int(num_epochs)

		# 1. Compute replication profiles for each replicate using chromosome 4
		from pipeline.fit_replication_profiles import main as fit_replication_profile
		fit_replication_profile(chrom=chrom, replicate=replicate, num_epochs=num_epochs, 
			output_directory=output_directory, from_CLOCCS=cold_start)


	# 2. Compute combined replication profiles for all chromosomes
	elif command == 'combined_replication':

		(_, command, output_directory) = system_args

		from src.CombinedReplicateDeconvolutionRunner import main as fit_combined_replication
		from src.config import load_default_configs

		# Load the replicate 1 and 2 configs from the individual runs, these should
		# have been fit on chromosome 4
		config1, config2 = load_config_from_replication_runs(out_dir, chrom=4)

		# For each chromosome, create the replication profiles for each of the chromosomes and save to disk
		for chrom in range(1, 17):
			combined_runner = fit_combined_replication(chrom, 1, out_dir, config1, config2)

	else:
		raise ValueError(f"Invalid command" + command)


	# Fit the cell cycle parameters using the previous individual replicate fits
	# as initial parameters
	# fit_combined_replication_profile(chrom=4, num_epochs=1)

	# Generate replication profiles for all chromosomes
	# generate_replication_profiles()





if __name__ == '__main__':
	main()
