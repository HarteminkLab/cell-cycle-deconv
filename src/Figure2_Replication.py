
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure_configs import FiguresConfig
from src.figure_configs import save_figure_for_paper
from src.plot_helpers import adjust_lightness_saturation


class Figure2ReplicationDeconvolution():
	"""Create figures for the replication deconvolution"""

	def __init__(self, output_directory):
		self.output_directory = output_directory
		pass
