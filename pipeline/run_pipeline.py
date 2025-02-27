

def main():

	# Run pipeline from start to finish

	output_directory = "output/pipeline/"

	# 1. Compute replication profiles for each replicate using chromosome 4
	from src.fit_replication_profiles import main as fit_replication_profile
	fit_replication_profile(chrom=4, replicate=1, num_epochs=1, output_directory=output_directory)
	fit_replication_profile(chrom=4, replicate=2, num_epochs=1, output_directory=output_directory)

	# 2. Compute combined replication profiles
	from src.fit_combined_replication_profiles import main as fit_combined_replication_profile

	# Fit the cell cycle parameters using the previous individual replicate fits
	# as initial parameters
	fit_combined_replication_profile(chrom=4, num_epochs=1)

	# Generate replication profiles for all chromosomes
	generate_replication_profiles()








if __name__ == '__main__':
	main()
