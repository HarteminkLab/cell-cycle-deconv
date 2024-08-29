
import numpy as np
import pandas as pd

from scipy.signal import find_peaks


def gaussian_kernel(size, sigma=1.0):
	"""Generate a 1D Gaussian kernel."""
	size = int(size)
	x = np.linspace(-size // 2, size // 2, size)
	kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
	return kernel / np.sum(kernel)


def filter_near_max_mins(important_points, min_distance):
	sorted_points = important_points.sort_values('value').reset_index(drop=True)

	minima = sorted_points[sorted_points.point_type == 'minima']
	maxima = sorted_points[sorted_points.point_type == 'maxima']

	drop_mins = []

	# Loop through minima and identify minima that are too close to one another
	# and store in a tuple
	for i in range(1, len(minima)):
		prev = minima.iloc[i-1]
		cur = minima.iloc[i]
		# If the pair are within the minimum distance, set to drop
		if cur.value-prev.value < min_distance:
			drop_mins.append((prev, cur))
	# Then iterate through the tuple and drop minima that are too close to one another
	# and the max in between
	for m1, m2 in drop_mins:
		# Drop the higher of the two
		drop_i = m1.name if m1.value > m2.value else m2.name
		sorted_points = sorted_points.drop(drop_i)

		# Drop the max that spans the two mins
		sorted_points = sorted_points.drop(m1.name+1)

	sorted_points = sorted_points.reset_index(drop=True)

	# --------- Repeat for the max values
	drop_maxes = []

	# Loop through maxima and identify maxima that are too close to one another
	# and store in a tuple
	for i in range(1, len(maxima)):
		prev = maxima.iloc[i-1]
		cur = maxima.iloc[i]
		# If the pair are within the minimum distance, set to drop
		if cur.value-prev.value < min_distance:
			drop_maxes.append((prev, cur))

	# Then iterate through the tuple and drop maxima that are too close to one another
	# and the max in between
	for m1, m2 in drop_maxes:
		# Drop the higher of the two
		drop_i = m1.name if m1.value > m2.value else m2.name
		sorted_points = sorted_points.drop(drop_i)

		# Drop the minima that spans the two maxima
		sorted_points = sorted_points.drop(m1.name+1)

	sorted_points = sorted_points.reset_index(drop=True)

	return sorted_points


def retrieve_important_points(expression_values, window_size=11, sigma=1.0, min_distance=10):
	"""Retrieve important points in a gene expression vector. 

	Smooth the data slightly then identify max and minimums. And points in which
	the slope is the greatest between these max and minima.
	"""
	# Generate Gaussian kernel
	kernel = gaussian_kernel(window_size, sigma)
	
	# Pad the expression values to handle edge effects
	padded_expression_values = np.concatenate([np.repeat(expression_values[0], window_size//2), expression_values,
		np.repeat(expression_values[-1], window_size//2)])

	# Smooth the data using Gaussian filter
	smoothed_data = np.convolve(padded_expression_values, kernel, mode='valid')
	
	# Find local minima and maxima
	minima, _ = find_peaks(-smoothed_data)
	maxima, _ = find_peaks(smoothed_data)
		
	smoothed_data_df = pd.DataFrame(smoothed_data)
	important_points = pd.concat([
		pd.DataFrame({'value': minima, 'point_type': ['minima']*len(minima)}),
		pd.DataFrame({'value': maxima, 'point_type': ['maxima']*len(maxima)}),
	])

	# Filter out max and mins that are too near
	important_points = filter_near_max_mins(important_points, min_distance)
	important_points.value = important_points.value.astype(int)

	# Calculate the first derivative (slope) of the smoothed data
	slope = np.gradient(smoothed_data)
  
	max_slope_points = []
	for i in range(1, len(important_points)):
		prev = important_points.iloc[i-1].value
		cur = important_points.iloc[i].value
		slopes_in_span = slope[prev:cur]

		max_slope_i = np.argmax(np.abs(slopes_in_span)) + prev
		max_slope_points.append(max_slope_i)

	# Compute the wrap around point
	prev = important_points.iloc[-1].value
	cur = important_points.iloc[0].value
	slope_wraparound_points = np.concatenate([slope[prev:], slope[:cur]])
	max_slope_i = np.argmax(np.abs(slope_wraparound_points)) + prev
	if max_slope_i > len(slope): max_slope_i = max_slope_i - len(slope)

	max_slope_points.append(max_slope_i)

	data_length = len(smoothed_data)

	important_points = pd.concat([important_points,
		pd.DataFrame({'value': max_slope_points, 'point_type': ['slope']*len(max_slope_points)}),
	])

	important_points = important_points.reset_index(drop=True)

	return smoothed_data_df, important_points, slope

