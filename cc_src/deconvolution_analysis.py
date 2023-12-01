
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DeconvolvedAnalysis:

	def __init__(self, model):
		"""

		For analysis we will want the g values, f values, H, (TODO: gamma values), interval defintions
		the g values live inside the model config.

		Have an ability to plot a single gene's profile (raw data, and deconvolved data), let's do this first
		Then, we will calculate the peak to trough ratios for every gene

		"""

		f_path = 'output/xin-deconvolved_fs.csv'

		self.model = model

		f = pd.read_csv(f_path)
		f.index = self.model.config.wt1_df.index

		self.f = f

