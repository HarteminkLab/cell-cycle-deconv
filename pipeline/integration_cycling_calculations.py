import numpy as np
import pandas as pd
from typing import Tuple


def calculate_trajectory_area(var1_data: np.ndarray, var2_data: np.ndarray) -> float:
	"""
	Calculate the area enclosed by the 2D trajectory using the shoelace formula.
	
	Parameters:
	var1_data: 1D array of variable 1 values across timepoints
	var2_data: 1D array of variable 2 values across timepoints
	
	Returns:
	Enclosed area of the trajectory
	"""
	if len(var1_data) != len(var2_data) or len(var1_data) < 3:
		return 0.0
	
	# Shoelace formula for polygon area
	n = len(var1_data)
	area = 0.0
	
	for i in range(n):
		j = (i + 1) % n  # Wrap around to close the polygon
		area += var1_data[i] * var2_data[j]
		area -= var2_data[i] * var1_data[j]
	
	return abs(area) / 2.0


def calculate_linkage_values(df1, df2, index) -> pd.DataFrame:
	"""
	Analyze cycling metrics for all genes across two datasets.
	
	Parameters:
	df1: DataFrame with genes as rows, timepoints as columns (variable 1)
	df2: DataFrame with genes as rows, timepoints as columns (variable 2)
	
	Returns:
	DataFrame with cycling metrics for each gene
	"""
	from scipy.stats import pearsonr, spearmanr

	results = []
	
	for i in range(len(index)):
		var1_data = df1[i]
		var2_data = df2[i]

		# Compute correlation measures
		pearsonr_value, pearson_p = pearsonr(var1_data, var2_data)
		spearmanr_value, spearman_p = spearmanr(var1_data, var2_data)
		traj_area = calculate_trajectory_area(var1_data, var2_data)
		
		results.append({
			'trajectory_area': traj_area,
			'pearsonr': pearsonr_value,
			'spearmanr': spearmanr_value,
			'pearsonr_p': pearson_p,
			'spearmanr_p': spearman_p,
		})
	
	return pd.DataFrame(results, index=index)
