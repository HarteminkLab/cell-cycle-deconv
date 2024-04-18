
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TimeDeltaAnalysis:
	"""This class will allow us to compute the time delta between two 
	measures of the cell cycle space. For example, we may want to know
	how much time elapses between a promoter TF binding event and gene
	expression.

	Note: We will keep in mind that the daughter and mother branches
	may need to be analyzed separately for simplicity sakes to start.
	"""
	def __init__(self, ge_analysis, promoter_analysis):
		self.ge_analysis = ge_analysis
		self.promoter_analysis = promoter_analysis
		
	def compute_max_mins_deltas(self):

		from src.timer import Timer
		timer = Timer()

		ge_analysis = self.ge_analysis
		promoter_analysis = self.promoter_analysis
		config = ge_analysis.config

		print("Computing max mins and deltas for mother and daughter")

		# For gene expression
		gene_fs = ge_analysis.gene_expression_f.astype(float)
		ge_mother_max_mins, ge_daughter_max_mins = compute_mother_daughter_max_mins(config, 
			gene_fs)

		# then for promotear occupancy
		gene_fs = promoter_analysis.sm_prom_occ_df.astype(float)
		prom_mother_max_mins, prom_daughter_max_mins = compute_mother_daughter_max_mins(config, 
			gene_fs)

		timer.print_time("Done.")

		mother_daughter_deltas = compute_mother_daughter_deltas(config, ge_mother_max_mins, prom_mother_max_mins,
									  ge_daughter_max_mins, prom_daughter_max_mins)


		self.ge_mother_max_mins = ge_mother_max_mins
		self.ge_daughter_max_mins = ge_daughter_max_mins

		self.prom_mother_max_mins = prom_mother_max_mins
		self.prom_daughter_max_mins = prom_daughter_max_mins
		self.mother_daughter_deltas = mother_daughter_deltas

	def plot_deltas(self):
		"""Plot the time deltas as a histogram. note: taking average of moether and daughters"""
		plt.figure(figsize=(4, 2))
		self.mean_deltas = self.mother_daughter_deltas.mean(axis=1)

		plt.hist(self.mean_deltas, bins=50)
		plt.title("Distribution of time delta between peak\ngene expression and promoter occupancy")
		plt.xlabel("Delta, minutes")

	def plot_scatter_ptr(self, thresholds=(1.2, 2.0), genes_mapping={}):

		plt.figure(figsize=(11, 4))

		plt.subplot(1, 2, 1)
		comparison_ptr_df = self.ge_analysis.ptrs_min_maxs[['ptr']].\
			join(self.promoter_analysis.promoter_min_maxs[['ptr']],
											   lsuffix='_ge', rsuffix='_prom')
		self.comparison_ptr_df = comparison_ptr_df

		# A border for low visibility points
		plt.scatter(comparison_ptr_df.ptr_ge, comparison_ptr_df.ptr_prom, s=3,
				   edgecolors='black', color='none')
		sc = plt.scatter(comparison_ptr_df.ptr_ge, comparison_ptr_df.ptr_prom, s=2,
				   c=self.mean_deltas, cmap='RdBu_r',
				vmin=-40, vmax=40)

		# Selected genes -------------
		geneset = self.ge_analysis.geneset

		from src.sgd import get_gene_name

		custom_modify_text_offset = {
			"PRY3": (0, -0.5),
			"CLB5": (0, -0.5)
		}
		text_offset = (0.01, 0.1)

		for color, selected_genes in genes_mapping.items():
			selected_orfs = geneset[geneset.gene.isin(selected_genes)].index.values

			selected_data = comparison_ptr_df.loc[selected_orfs]
			plt.scatter(selected_data.ptr_ge, selected_data.ptr_prom, 
				edgecolors=color, marker='D', s=20, lw=0.5, facecolors='none')

			thresholded_selected = selected_data[(selected_data.ptr_ge > thresholds[0]) & 
												 (selected_data.ptr_prom > thresholds[1])]

			for orf_name, row in thresholded_selected.iterrows():
				gene_name = get_gene_name(orf_name)
				if gene_name is None: gene_name = orf_name

				x, y = row.ptr_ge+text_offset[0], row.ptr_prom+text_offset[1]
				if gene_name in custom_modify_text_offset.keys():
					x += custom_modify_text_offset[gene_name][0]
					y += custom_modify_text_offset[gene_name][1]

				plt.text(x, y, gene_name)

		# ----------------------------

		plt.axvline(thresholds[0], c='black', lw=1, ls='dotted')
		plt.axhline(thresholds[1], c='black', lw=1, ls='dotted')

		plt.xlabel("Gene expression PTR")
		plt.ylabel("Promoter occupancy PTR")
		cbar = plt.colorbar(sc)
		plt.title("Distribution of gene expression\nand promoter occupancy PTRs")

		# Does the distribution of deltas change with the thresholded data?
		plt.subplot(1, 2, 2)

		thresholded_dat = comparison_ptr_df[(comparison_ptr_df.ptr_ge > thresholds[0]) & 
			(comparison_ptr_df.ptr_prom > thresholds[1])]
		k = len(thresholded_dat)
		plt.hist(self.mean_deltas.loc[thresholded_dat.index], bins=30)
		plt.title(f"Thresholded time deltas\nn={k}")
		plt.xlabel("Time delta, minutes")


	def plot_gene_trace(self, orf_name):

		from src.TracerPlotter import TracePlotter
		from src.sgd import get_gene_title_name

		tracer_plotter = TracePlotter(self.ge_analysis, self.promoter_analysis)

		row = self.mother_daughter_deltas.loc[orf_name]

		gene_title = get_gene_title_name(orf_name)

		title = f"{gene_title}\nMother delta: {row.delta_min_mother:.1f}', Daughter delta: {row.delta_min_daughter:.1f}'"

		tracer_plotter.plot_time_delta_curves(orf_name, title=title)


def compute_mother_daughter_max_mins(config, gene_fs):
	"""
	Compute the mother daughter max min values given a df of F metrics
	
	gene_fs: Either gene expression or a single value of a chromatin per gene per deconvolved
	timepoints. e.g. (5500 x 227)
	
	return the mother max min dataframe, daughter max min dataframe
	"""
	from src.peak_to_trough import compute_max_min_locations

	min_max_arr = np.apply_along_axis(lambda row: compute_max_min_locations(config, row, 
		ret_all=True), 1, gene_fs)
	min_max_arr = min_max_arr[:, :, :, :3].astype(float)

	mother_branch_mins = min_max_arr[:, 1, 0]
	mother_branch_maxs = min_max_arr[:, 1, 1]

	daughter_branch_mins = min_max_arr[:, 2, 0]
	daughter_branch_maxs = min_max_arr[:, 2, 1]

	def get_max_min_df(dat):
		mins_df = pd.DataFrame(columns=['value', 'index', 'tp'], data=dat[:, 0], index=gene_fs.index)
		maxs_df = pd.DataFrame(columns=['value', 'index', 'tp'], data=dat[:, 1], index=gene_fs.index)
		return mins_df.join(maxs_df, lsuffix='_min', rsuffix='_max')
	
	mother_max_mins = get_max_min_df(min_max_arr[:, 1])
	daughter_max_mins = get_max_min_df(min_max_arr[:, 2])

	return mother_max_mins, daughter_max_mins


def get_delta_angle_differences(config, ge_mother_max_mins, prom_mother_max_mins, branch):
	""""Compute the minimal angle between the max values for gene expression and promoter occupancy

	Note: Ambiguity in the word "min" refers to minimal angle. Whereas we are computing the maximum 
	value: gene expression and promoter occupancy (or chromatin metric value).
	"""

	t_tps = config.get_timepoints_for_branch(branch)

	delta_angles_df = ge_mother_max_mins[['tp_max']].join(prom_mother_max_mins[['tp_max']], 
		lsuffix='_ge', rsuffix='_prom')
	ge_angle = delta_angles_df.tp_max_ge.apply(lambda tp: 
											   convert_to_angle_deg(t_tps, tp))
	prom_angle = delta_angles_df.tp_max_prom.apply(lambda tp: 
												   convert_to_angle_deg(t_tps, tp))
	delta_angles_df['ge_angle_deg'] = ge_angle
	delta_angles_df['prom_angle_deg'] = prom_angle

	minimal_degrees = minimal_angle_degrees(delta_angles_df.ge_angle_deg, 
		delta_angles_df.prom_angle_deg)

	delta_angles_df['delta_deg'] = minimal_degrees
	delta_angles_df['delta_min'] = convert_angle_to_mins(t_tps, delta_angles_df.delta_deg)

	return delta_angles_df


def compute_mother_daughter_deltas(config, ge_mother_max_mins, prom_mother_max_mins,
	ge_daughter_max_mins, prom_daughter_max_mins):
	mother_deltas_df = get_delta_angle_differences(config, ge_mother_max_mins, 
												   prom_mother_max_mins, 't')

	daughter_deltas_df = get_delta_angle_differences(config, ge_daughter_max_mins, 
													 prom_daughter_max_mins, 'b')
	mother_daughter_max_deltas = mother_deltas_df[['delta_min']].join(
		daughter_deltas_df[['delta_min']],
										lsuffix='_mother', rsuffix='_daughter')
	return mother_daughter_max_deltas


# def minimal_angle_degrees(angle1, angle2):
# 	# Compute the absolute difference and map it into the range [0, 360]
# 	difference = np.abs(angle1 - angle2) % 360
	
# 	# If the difference is greater than 180 degrees, take the shorter way around the circle
# 	difference[difference > 180] = 360 - difference



# 	#difference[difference > 180] = difference-180
	
# 	return difference

def minimal_angle_degrees(angle1, angle2):
	# Calculate the difference
	difference = angle1 - angle2
	# Normalize the difference to -180 to 180 range
	normalized_difference = (difference + 180) % 360 - 180
	# Adjust to ensure if it's over 180, it wraps around to give a value between -180 and 180
	normalized_difference[normalized_difference > 180] = normalized_difference[normalized_difference > 180] - 360

	return normalized_difference

def convert_angle_to_mins(t_tps, angle):
	len_tps = t_tps.max() - t_tps.min()

	# Map the angle back to cell cycle minute space
	# originally the angle was computed by the length of the cell cycle
	# so let's convert it back
	proportion = angle / 360.
	time_mins = proportion*len_tps

	return time_mins


def convert_to_angle_deg(t_tps, tp):
	min_tp, max_tp = t_tps.min(), t_tps.max()
	proportion = (tp - min_tp) / (max_tp-min_tp)
	return proportion * 360.
