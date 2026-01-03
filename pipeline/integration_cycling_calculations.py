import numpy as np
import pandas as pd
from typing import Tuple

def calculate_perimeter(x, y):
	"""
	Calculate perimeter of trajectory defined by x, y coordinates.
	Assumes points are in chronological order.
	"""
	# Calculate distances between consecutive points
	dx = np.diff(x)
	dy = np.diff(y)
	segment_lengths = np.sqrt(dx**2 + dy**2)
	
	# Sum all segments, including closing segment back to start
	perimeter = np.sum(segment_lengths)
	
	# Add closing segment (last point back to first)
	closing_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
	perimeter += closing_distance
	
	return perimeter
	
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

def calculate_diameter_squared(var1_data: np.ndarray, var2_data: np.ndarray) -> float:
	"""Calculate the square of the maximum distance between any two points."""
	if len(var1_data) < 2:
		return 0.0
	
	max_dist_sq = 0.0
	for i in range(len(var1_data)):
		for j in range(i + 1, len(var1_data)):
			dist_sq = (var1_data[j] - var1_data[i])**2 + (var2_data[j] - var2_data[i])**2
			max_dist_sq = max(max_dist_sq, dist_sq)
	
	return max_dist_sq


def calculate_bounding_box_area(var1_data: np.ndarray, var2_data: np.ndarray) -> float:
	"""Calculate the area of the bounding rectangle."""
	var1_range = np.max(var1_data) - np.min(var1_data)
	var2_range = np.max(var2_data) - np.min(var2_data)
	return var1_range * var2_range


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
		
		# Normalization scalars
		diameter_sq = calculate_diameter_squared(var1_data, var2_data)
		bounding_box_area = calculate_bounding_box_area(var1_data, var2_data)

		# Pseudocount prevents small numbers from overinflating the normalized
		# area
		eps = 1

		results.append({
			'trajectory_area': traj_area,
			'normalized_trajectory_area': traj_area/(diameter_sq+eps),
			'diameter_sq': diameter_sq,
			'bounding_box_area': bounding_box_area,
			'pearsonr': pearsonr_value,
			'spearmanr': spearmanr_value,
			'pearsonr_p': pearson_p,
			'spearmanr_p': spearman_p,
		})
	
	return pd.DataFrame(results, index=index)
