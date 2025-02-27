
import sys
sys.path.append('.')

import sys
from src.utils import mkdirs_safe, parse_bool


def main():

	system_args = tuple(sys.argv)
	command = system_args[1]

	# Run pipeline from start to finish

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

	else:
		raise ValueError(f"Invalid command" + command)

	# 2. Compute combined replication profiles
	# from src.fit_combined_replication_profiles import main as fit_combined_replication_profile

	# Fit the cell cycle parameters using the previous individual replicate fits
	# as initial parameters
	# fit_combined_replication_profile(chrom=4, num_epochs=1)

	# Generate replication profiles for all chromosomes
	# generate_replication_profiles()





if __name__ == '__main__':
	main()
