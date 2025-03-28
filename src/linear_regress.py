import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def fit_linear_regression(x, y):
	"""
	Fit a linear regression model to x and y data.
	
	Parameters:
	-----------
	x : array-like
		Independent variable values
	y : array-like
		Dependent variable values
	
	Returns:
	--------
	dict : Dictionary containing regression results with keys:
		- slope: Slope of the regression line
		- intercept: Intercept of the regression line
		- r2: R-squared value
		- p_value: p-value for the slope
		- std_err: Standard error of the slope
		- rmse: Root mean squared error
	"""
	# Ensure input is numpy array
	x = np.asarray(x)
	y = np.asarray(y)
	
	# Perform regression using scipy.stats
	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
	r2 = r_value**2
	
	# Calculate predictions
	y_pred = slope * x + intercept
	
	# Calculate RMSE
	rmse = np.sqrt(np.mean((y - y_pred)**2))
	
	# Return results as a dictionary
	results = {
		'slope': slope,
		'intercept': intercept,
		'r2': r2,
		'p_value': p_value,
		'std_err': std_err,
		'rmse': rmse
	}
	
	return results
