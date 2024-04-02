
from matplotlib import pyplot as plt
import numpy as np
import scipy


def plot_phase_stack(tp, prev_vec, cur_vec, name):
	from src.model import color_for_key

	if prev_vec is None:
		prev_vec = np.zeros_like(cur_vec)
	
	cur_stack = prev_vec+cur_vec
	
	plt.fill_between(tp, prev_vec, cur_stack, color=color_for_key(name), label=name)

	return cur_stack

	
def plot_H_as_growth_curve(model):

	plt.figure(figsize=(8, 2))
	plt.subplot(1, 2, 1)
	plt.imshow(model.H, aspect='auto', vmax=0.02, cmap='viridis')
	from src.plot_helpers import plot_H_as_growth_curve

	plt.subplot(1, 2, 2)
	tp = model.config.WT1_TIMEPOINTS
	H = model.H[:len(tp)]

	phase_columns = model.config.phase_columns
	keys = list(phase_columns.keys())

	plt.ylim(0, 2.)

	phases = []
	vectors = []

	for phase, indices in phase_columns.items():
		cur_vec = H[:, indices].sum(axis=1)
		vectors.append(cur_vec)
		phases.append(phase)
		
	plot_stacked_curves(tp, vectors, phases)

	plt.legend()


def plot_stacked_curves(x, vectors, names):
	
	for i in range(len(vectors)):

		if i == 0:
			previous_stack = None
	
		cur_vector = vectors[i]
		name = names[i]

		previous_stack = plot_phase_stack(x, prev_vec=previous_stack, 
										  cur_vec=cur_vector, name=name)
	
def plot_density(data, ax=None, color='red', arange=None, 
	alpha=1., zorder=1, fill=False, bw=10, neg=False, 
	mult=1.0, y_offset=0, flip=False, lw=1, label=None, ls='solid'):

	from sklearn.neighbors import KernelDensity
	def _kde_sklearn(x, x_grid, bandwidth):
		kde_skl = KernelDensity(bandwidth=bandwidth)
		kde_skl.fit(x[:, np.newaxis])
		log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
		pdf = np.exp(log_pdf)
		return pdf

	if ax is None:
		fig, ax = plt.subplots()

	if arange is None:
		arange = min(data), max(data), 1

	x = np.arange(arange[0], arange[1], arange[2])

	y = _kde_sklearn(data, x, bw) * mult
	d = scipy.zeros(len(y))
	fill_mask = y >= d

	if fill:
		if not flip:
			ax.fill_between(x, y+y_offset, 0, color=color,
					 alpha=alpha, linewidth=1, zorder=zorder, ls=ls)
		else:
			ax.fill_betweenx(x, y+y_offset, 0, color=color,
					 alpha=alpha, linewidth=1, zorder=zorder, ls=ls)
	else:
		if not flip:
			ax.plot(x, y+y_offset, color=color,
				 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
				 solid_joinstyle='round', ls=ls)
		else:
			ax.plot(y+y_offset, x, color=color,
				 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
				 solid_joinstyle='round', ls=ls)

	return y