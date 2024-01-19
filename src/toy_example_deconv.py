
import numpy as np
import matplotlib.pyplot as plt


class ToyDeconv:


	def __init__(self, g1, g2):


		# Simple 2x2 with easy to compute numbers
		# enforce each row must sum to 1
		# i.e. each row is the cell's popluation in different portions of the cell cycle

		# Have variables for H for derivative calculations
		a, b, c, d = 0.25, 0.75, \
					 0.75, 0.25
		H = np.array([ [a, b],
					   [c, d]])

		# Our gene expression values
		g = np.array([[g1],
					  [g2]])

		self.a, self.b, self.c, self.d = a, b, c, d
		self.H = H
		self.g = g
		self.g1 = g1
		self.g2 = g2


	def compute_l2_sq_norm(self, f1, f2):
		"""
		Objective:
		 Minimize with respect to f:
		
		  ||  Hf - g  || (l2 norm)
		
		"""

		a, b, c, d = self.a, self.b, self.c, self.d
		g1 = self.g1
		g2 = self.g2

		f = [f1, f2]
		
		# The squared l2 norm (squared or unsquared won't offect where
		# the solution is)
		l2_squared_norm = (a*f1 + b*f2 - g1)**2 + (c*f1 + d*f2 - g2)**2

		return l2_squared_norm


	def find_solution(self, log=False):

		a, b, c, d = self.a, self.b, self.c, self.d
		u = self.g1
		v = self.g2

		ddx_coeff_y = (2*a*b + 2*c*d)
		ddx_coeff_x = (2*a*a+2*c*c)
		ddx_intercept = (2*a*u + 2*c*v)

		if log:
			print("Take the partial derivative d/dx")
			print(f"\n\t{ddx_coeff_x}x + {ddx_coeff_y}y = {ddx_intercept}")

		ddy_coeff_y = (2*b*b + 2*d*d)
		ddy_coeff_x = (2*a*b+2*c*d)
		ddy_intercept = (2*b*u + 2*d*v)

		if log:
			print("\nTake the partial derivative d/dy")
			print(f"\n\t{ddy_coeff_x}x + {ddy_coeff_y}y = {ddy_intercept}")

		# Now that we have a system of equations from the partial derivatives
		# We can solve for x and y to determine the values of x and y that minimize the 
		# L2 norm

		"""
		Rearrange so y is on the left, then set equal to one another:

		   (ddx_intercept - ddx_coeff_x * x) / ddx_coeff_y
		   = (ddy_intercept - ddy_coeff_x * x) / ddy_coeff_y

		Solve for x:

		ddx_intercept/ddx_coeff_y - ddy_intercept/ddy_coeff_y +
		 [(-ddx_coeff_x/ddx_coeff_y) + (ddy_coeff_x/ddy_coeff_y)]*x = 0


		-(ddx_intercept/ddx_coeff_y - ddy_intercept/ddy_coeff_y) /
		 [(-ddx_coeff_x/ddx_coeff_y) + (ddy_coeff_x/ddy_coeff_y)]
		"""

		solution_x = -(ddx_intercept/ddx_coeff_y - ddy_intercept/ddy_coeff_y) / \
		 ((-ddx_coeff_x/ddx_coeff_y) + (ddy_coeff_x/ddy_coeff_y))

		solution_y = (ddx_intercept - ddx_coeff_x*solution_x) / ddx_coeff_y

		if log:
			print("\nThus, the solution to the optimization is:\n")
			print(f"\tx = {solution_x:.2f}\n\ty = {solution_y:.2f}")

		self.solution = (solution_x, solution_y)

	def plot_optimization_and_solution(self):

		f1 = np.linspace(-5, 5, 100)
		f2 = np.linspace(-5, 5, 100)

		F1, F2 = np.meshgrid(f1, f2)
		Z = np.zeros_like(F1)

		for i in range(Z.shape[0]):
		    for j in range(Z.shape[1]):
		        Z[i][j] = self.compute_l2_sq_norm(F1[i][j], F2[i][j])


		# Create a figure for plotting
		fig = plt.figure(figsize=(8, 6))
		ax = plt.gca()

		# Plot the 3D contour plot
		contour = ax.contourf(F1, F2, Z, 1000, cmap='Spectral', vmax=10)
		plt.colorbar(contour)

		plt.axvline(0, c='black', lw=1, linestyle="dashed")
		plt.axhline(0, c='black', lw=1, linestyle="dashed")

		plt.scatter(self.solution[0], self.solution[1], s=30, color='white', marker='D')
		plt.text(self.solution[0], self.solution[1], f"  {self.solution[0]:.2g}, {self.solution[1]:.2g}", c='white',
			ha='left', va='top')

		plt.xlabel("$f_1$")
		plt.ylabel("$f_2$")
