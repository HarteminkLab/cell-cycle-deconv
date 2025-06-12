

# Deprecated commands
# Chromatin and expression are largely independent, so these analyses and files will be

# 1. Deconvolve individual replication profiles, learn cell cycle parameters from MNase-seq
# Now, just computing the combied replication profile with no cell cycle parameter updating

elif command == 'replication':

	# Deprecated, we'll go straight into the combined replication model without
	# parameter searching
	pass

	# (_, command, output_directory, replicate, chrom, num_epochs, cold_start) = system_args

	# single_replication_directory = f"{output_directory}/single_replication"
	# mkdirs_safe([output_directory, single_replication_directory])

	# cold_start = parse_bool(cold_start) # Unused parameter, may deprecate
	# chrom = int(chrom)
	# replicate = int(replicate)
	# num_epochs = int(num_epochs)

	# # 1. Compute replication profiles for each replicate using chromosome 4
	# from pipeline.fit_replication_profiles import main as fit_replication_profile
	# fit_replication_profile(chrom=chrom, replicate=replicate, num_epochs=num_epochs, 
	# 	output_directory=single_replication_directory)

# restructured to focus on the chromatin first.

	elif command == 'expression_chromatin_analysis':

		from pipeline.expression_chromatin_analysis_runner import ExpressionChromatinAnalysis
		(_, command, output_directory) = system_args

		# Create analysis object
		analyzer = ExpressionChromatinAnalysis(output_directory)

		# Run all analyses
		analyzer.run_all_analyses()

		print("Analysis completed successfully")

		from pipeline.expression_chromatin_analysis_runner import layout_figure_plots

		plots_dir = f'{output_directory}/expression_chromatin'
		save_dir = plots_dir

		print("Laying out plots for Figure 2")
		layout_figure_plots(plots_dir, save_dir)

		print("Laying out plots for Supplemental figures")
		layout_supplemental_1(plots_dir, save_dir)
		layout_supplemental_2(plots_dir, save_dir)
		layout_supplemental_3(plots_dir, save_dir)


	# todo: Currently, using partial daughter as the baseline model

	# elif command == 'deconvolve_chromatin_staging':

	# 	(_, command, output_directory, index) = system_args
	# 	chromatin_save_directory = f"{output_directory}/chromatin_deconvolution_staging/"
	# 	index = int(index)

	# 	window_set_path = "datasets/computed_mnase/test_window_set_2kb.csv"
	# 	deconvolve_chromatin(chromatin_save_directory, window_set_path, index)

	# elif command == 'deconvolve_chromatin_full':

	# 	(_, command, output_directory, index) = system_args
	# 	chromatin_save_directory = f"{output_directory}/chromatin_deconvolution/"
	# 	mkdirs_safe([chromatin_save_directory])
	# 	index = int(index)

	# 	window_set_path = "data/reference_data/sacCer3_genome_10k_windows.csv"
	# 	deconvolve_chromatin(chromatin_save_directory, window_set_path, index)


# Deprecated
	# elif command == 'replication_second_stage':

	# 	(_, command, output_directory, replicate, chrom, num_epochs, cold_start) = system_args

	# 	single_replication_directory = f"{output_directory}/single_replication"

	# 	cold_start = parse_bool(cold_start)
	# 	chrom = int(chrom)
	# 	replicate = int(replicate)
	# 	num_epochs = int(num_epochs)

	# 	# 1. Compute replication profiles for each replicate using chromosome 4
	# 	from pipeline.fit_replication_profiles import main as fit_replication_profile
	# 	fit_replication_profile(chrom=chrom, replicate=replicate, num_epochs=num_epochs, 
	# 		output_directory=single_replication_directory, cold_start=cold_start)
