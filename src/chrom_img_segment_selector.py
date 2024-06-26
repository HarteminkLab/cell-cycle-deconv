
from src.global_config import GlobalConstants
import numpy as np


def translate_span_for_bins(center_pos, search_span, flip=False):
	"""

	todo: This function does multiple things that may be worth separating out into 
	additional logic, currently it is only used for nucleosome tracking for origins.
	but will anticipate using this for capturing gene body nucleosomes and small fragment
	windows.

	Given a search span and center position, tranlsate the search span to a span
	that is rounded to the nearest bin.

	Then adds the search span to the center position to return a genomic range
	that the search span represents in the genomic coordinate space.

	Flip if necessary for crick strand positions
	"""
	
	def round_nearest_bin(x, bin_width=GlobalConstants.BIN_WIDTH):
		return int(round(x/bin_width)) * bin_width

	if flip:
		search_span = -search_span[1],\
			-search_span[0]

	search_span = round_nearest_bin(search_span[0]),\
		round_nearest_bin(search_span[1])
	ret_span = center_pos + search_span[0], center_pos \
		+ search_span[1]

	return ret_span

