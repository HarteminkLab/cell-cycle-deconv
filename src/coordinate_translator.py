
import numpy as np


class CoordinateTranslator():
	"""Small class to perform coordinate system transforms in linear space

	This is useful for the genomic position bins that map to nparrays that do not have
	knowledge of the original genomic positions in their indices.


	Example:
	A genomic range of 2000-3000 and an array of 0-1000
	"""
	def __init__(self, span):
		self.span = span

	
	def translate(self, x, new_origin=0):
		"""Translate a value in this space to a new space, origin=0 is useful for np.arrays with
		0-indexing"""
		return x-self.span[0]


