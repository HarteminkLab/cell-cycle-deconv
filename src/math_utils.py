import numpy as np
from scipy.interpolate import interp1d

# 4. Define limit calculation helper
def create_data_lims(data, padding):
	"""
	Compute axis limits from data with padding.
		
	Returns
	-------
	tuple
		(lower_limit, upper_limit)
	"""
	min_max = np.quantile(data, q=[0, 1])
	value_range = min_max[1] - min_max[0]
	ret = (min_max[0] - padding * value_range,
			min_max[1] + padding * value_range)
	return ret
		

def compute_signed_area(x, y):
	"""
	Compute signed area of a closed curve using shoelace formula.
	Positive = CCW, Negative = CW
	"""
	signed_area = 0
	n = len(x)
	for i in range(n - 1):
		signed_area += (x[i] * y[i + 1] - x[i + 1] * y[i])
	# Close the loop
	signed_area += (x[-1] * y[0] - x[0] * y[-1])
	return signed_area / 2

def compute_offset_curve(x, y, offset_distance_x, offset_distance_y, outward=True):
	"""
	Compute an offset curve parallel to the input curve.
	
	Parameters:
	-----------
	x, y : array-like
		Coordinates of the original curve
	offset_distance : float
		Distance to offset (positive value)
	outward : bool
		If True, offset away from the curve interior (based on signed area)
		If False, offset toward the interior
	
	Returns:
	--------
	x_offset, y_offset : arrays
		Coordinates of the offset curve
	"""
	x = np.array(x)
	y = np.array(y)
	n = len(x)
	
	# Determine curve orientation
	signed_area = compute_signed_area(x, y)
	is_ccw = signed_area > 0
	
	# Initialize offset coordinates
	x_offset = np.zeros(n)
	y_offset = np.zeros(n)
	
	for i in range(n):
		# Compute tangent vector using central differences
		if i == 0:
			# Forward difference at start
			dx = x[i + 1] - x[i]
			dy = y[i + 1] - y[i]
		elif i == n - 1:
			# Backward difference at end
			dx = x[i] - x[i - 1]
			dy = y[i] - y[i - 1]
		else:
			# Central difference in middle
			dx = x[i + 1] - x[i - 1]
			dy = y[i + 1] - y[i - 1]
		
		# Normalize tangent vector
		length = np.sqrt(dx**2 + dy**2)
		if length > 0:
			dx /= length
			dy /= length
		
		# Compute normal vector (perpendicular to tangent)
		# Rotate tangent 90 degrees
		# Right-hand normal: (-dy, dx)
		# Left-hand normal: (dy, -dx)
		
		# Determine which normal to use based on orientation and outward preference
		if outward:
			# For CCW curves, right-hand normal points outward
			# For CW curves, left-hand normal points outward
			if is_ccw:
				nx, ny = dy, -dx  # Right-hand normal
			else:
				nx, ny = -dy, dx  # Left-hand normal
		else:
			raise ValueError("Unimplemented")
		
		# Apply offset
		x_offset[i] = x[i] + offset_distance_x * nx
		y_offset[i] = y[i] + offset_distance_y * ny
	
	return x_offset, y_offset


def offset_curve_shapely(x, y, distance=0.1):
	from shapely import LineString
	"""
	Create an offset curve using Shapely's buffer boundary method.
	
	Parameters:
	-----------
	x, y : array-like
		Input curve coordinates
	distance : float
		Offset distance (positive = expand/outward)
		
	Returns:
	--------
	offset_x, offset_y : numpy arrays
		Coordinates of the offset curve boundary
	"""
	# Create LineString from input points
	line = LineString(np.column_stack([x, y]))
	
	# Create buffer polygon
	# cap_style=2 (flat), join_style=2 (mitre)
	buffer_poly = line.buffer(distance, cap_style=2, join_style=2)
	
	# Extract the boundary coordinates
	offset_x, offset_y = buffer_poly.exterior.xy

	# Get offset boundary coordinates
 	# Remove duplicate closing point
	offset_coords = np.array(buffer_poly.exterior.coords[:-1]) 

	# Find the closest point on offset to original start point
	start_point = np.array([x[0], y[0]])
	distances = np.linalg.norm(offset_coords - start_point, axis=1)
	start_idx = np.argmin(distances)
	
	# Rotate the offset coordinates to start at the aligned index
	offset_coords_aligned = np.roll(offset_coords, -start_idx, axis=0)
	
	return offset_coords_aligned[:, 0], offset_coords_aligned[:, 1]

	# return np.array(offset_x), np.array(offset_y)


def ordinary_least_squares(x, y):
	"""OLS regression"""
	x_centered = x - np.mean(x)
	y_centered = y - np.mean(y)

	if np.sum(x_centered) == 0:
		return np.nan, np.nan

	slope = np.sum(x_centered * y_centered) / np.sum(x_centered**2)
	intercept = np.mean(y) - slope * np.mean(x)
	return slope, intercept

