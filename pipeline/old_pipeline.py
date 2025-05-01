

# Deprecated commands
# Chromatin and expression are largely independent, so these analyses and files will be
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
