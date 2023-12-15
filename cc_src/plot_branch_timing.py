
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
import numpy as np

COLOR_MAP = {'S': [0.56862745, 0.70588235, 0.38431373],
		 'Gr': [0.78039216, 0.58039216, 0.56470588],
		 'G1': [0.57647059, 0.65882353, 0.77647059],
		 'DG1': [0.64705882, 0.77254902, 0.8       ],
		 'G2/M': [0.8745098 , 0.75294118, 0.61960784],
		 'Gd': np.array([201, 159, 249])/255.}

Y_GEN_JUMP = 0.5
DIV_W_2 = 0.25
G_R_MAP = {
	(0, 0): 0,
	(1, 1): -Y_GEN_JUMP*2,
	(1, 2): -Y_GEN_JUMP,
	(2, 2): -Y_GEN_JUMP*3,
}


class BranchPlotter:

	def __init__(self):

		self.recovery_time = 21
		self.lambd = 79.3787

		self.gamma1 = 0.157
		self.gamma2 = 0.4

		self.g1 = self.lambd*self.gamma1
		self.s = self.lambd*self.gamma2-self.g1
		self.g2m = self.lambd-self.g1-self.s
		
		self.daughter_delay = 1.1588

	def plot_branch_interval(self, ax, name, x1, x2, y):
		ax.plot([x1, x2], [y, y], lw=20, solid_capstyle='butt', c=COLOR_MAP[name])
		ax.text(x1/2+x2/2, y, name, fontsize=16, va='center', ha='center', c='white')

	def plot_cohort_cell_cycle(self, ax, g_r, cell_cycle_num):
		
		recovery_time = self.recovery_time
		g1 = self.g1
		s = self.s
		g2m = self.g2m
		lambd = self.lambd
		daughter_delay = self.daughter_delay

		g, r = g_r

		# Offset for where the start of G1 should be plotted for the cell cycle
		# daughter cells will be offset by the daughter specific delay
		start_offset = self.start_offset_for(g_r, cell_cycle_num)

		self.plot_cohort_offset(ax, g_r, cell_cycle_num, start_offset)

	def start_offset_for(self, g_r, cell_cycle_num):

		g, r = g_r
		recovery_time = self.recovery_time
		g1 = self.g1
		s = self.s
		g2m = self.g2m
		lambd = self.lambd
		daughter_delay = self.daughter_delay

		if g == 0:
			offset = (recovery_time + s + g2m)*(cell_cycle_num > 0)+lambd*(cell_cycle_num > 0)*(cell_cycle_num-1)
		else:
			offset = (recovery_time + s + g2m) + lambd*(r-1) \
					 + (lambd+daughter_delay)*(cell_cycle_num > 0) \
					 + self.daughter_delay*(r-1)*(g > 1)

		return offset


	def plot_cycle(self, ax, g_r, offset, cell_cycle_num):

		g, r = g_r
		recovery_time = self.recovery_time
		g1 = self.g1
		s = self.s
		g2m = self.g2m
		lambd = self.lambd
		daughter_delay = self.daughter_delay

		if (g, r) in G_R_MAP.keys():
			y = G_R_MAP[(g, r)]
		else: y = -10

		mapping = {
			'Gr': recovery_time,
			'Gd': daughter_delay,
			'G1': g1,
			'S': s,
			'G2/M': g2m
		}

		if g == 0 and r == 0 and cell_cycle_num == 0:
			boundaries = ['Gr', 'S', 'G2/M']
		elif cell_cycle_num > 0:
			boundaries = ['G1', 'S', 'G2/M']
		elif g > 0 and cell_cycle_num == 0:
			boundaries = ['Gd', 'G1', 'S', 'G2/M']
		else:
			return

		prev_x = offset
		for phase in boundaries:
			next_x = prev_x+mapping[phase]
			self.plot_branch_interval(ax, phase, prev_x, next_x, y)
			prev_x = next_x


	def plot_cohort_offset(self, ax, g_r, cell_cycle_num, start_offset):

		g, r = g_r
		recovery_time = self.recovery_time
		g1 = self.g1
		s = self.s
		g2m = self.g2m
		lambd = self.lambd
		daughter_delay = self.daughter_delay

		offset = start_offset

		self.plot_cycle(ax, g_r, offset, cell_cycle_num)


	def plot_mass(self, ax, experimental_time, translate=False):

		recovery_time = self.recovery_time
		g1 = self.g1
		s = self.s
		g2m = self.g2m
		lambd = self.lambd
		daughter_delay = self.daughter_delay
		y_offset = 0.05

		plot_time = experimental_time

		for g_r in G_R_MAP.keys():

			y_pos = G_R_MAP[g_r]
			meanG = plot_time
			std = 5
			g, r = g_r

			start_offset = self.start_offset_for(g_r, 0)

			lin_start, lin_end = meanG-std*3, meanG+std*3

			if lin_end < start_offset:
			   continue

			if g_r[0] > 0:
				start_offset

			lin_start = max(lin_start, start_offset)

			x = np.linspace(lin_start, lin_end, 1000)

			mass = norm(loc=meanG, scale=std).pdf(x)

			if translate:
				if g > 0:
					translated = self.start_offset_for(g_r, 0)
					x = x-translated-self.daughter_delay
				else:
					x = x-self.recovery_time+self.g1

			ax.fill_between(x, mass+y_offset+y_pos, y_offset+y_pos, color='red')


	def plot_full_branch(self, ax1, t):

		recovery_time = self.recovery_time
		g1 = self.g1
		s = self.s
		g2m = self.g2m
		lambd = self.lambd
		daughter_delay = self.daughter_delay

		ax1.set_ylim(-2, 0.5)

		xticks = np.arange(-100, 300, 20)

		ax1.set_xticks(xticks)
		ax1.set_xlim(-50, 300)

		ax1.set_xlabel("Experimental time, minutes", fontsize=16)

		ax1.axvline(0, c='#ddd', linestyle='solid', lw=1, zorder=0)

		self.plot_cohort_cell_cycle(ax1, (0, 0), 0)
		self.plot_cohort_cell_cycle(ax1, (0, 0), 1)
		self.plot_cohort_cell_cycle(ax1, (0, 0), 2)

		self.plot_cohort_cell_cycle(ax1, (1, 1), 0)
		self.plot_cohort_cell_cycle(ax1, (1, 1), 1)

		self.plot_cohort_cell_cycle(ax1, (1, 2), 0)
		self.plot_cohort_cell_cycle(ax1, (2, 2), 0)

		connections = [
			((0, 0), (1, 1)),
			((0, 0), (1, 2)),
			((1, 1), (2, 2)),
		]

		for s, e in connections:
			y1, y2 = G_R_MAP[s], G_R_MAP[e]
			x = self.start_offset_for(e, 0)
			ax1.plot([x, x], [y1+0.09, y2-0.09], c='black', lw='2')

		self.set_yticks(ax1)


	def set_yticks(self, ax):
		yticks = []
		yticklabels = []
		for k, v in G_R_MAP.items():
			yticks.append(v)
			yticklabels.append(k)
			
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticklabels)
		ax.set_ylabel("Cohort, {generation, reproductive instance}", fontsize=16)


	def plot_figure(self, t):

		fig = plt.figure(figsize=(16, 16))  # Create a figure with the desired size
		fig.tight_layout(rect=[0.0, 0., 1.0, 1.0])

		# Create ax1 using subplots2grid
		ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)  # Takes full width and top 75% (3 rows)

		# Create ax2 using subplots2grid
		ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)  # Takes full width and bottom 25% (1 row)

		ax1.set_title(f"{t} minutes", fontsize=20)

		from matplotlib import rcParams
		rcParams['axes.titlepad'] = 20

		plt.subplots_adjust(hspace=0.75, wspace=0.0)

		self.plot_full_branch(ax1, t)
		self.plot_gen(ax2)

		self.plot_mass(ax1, t)
		self.plot_mass(ax2, t, translate=True)

		return fig

	def translated_offset(self, g_r, cell_cycle_num):
		g, r = g_r
		g_r_0_offset = self.start_offset_for(g_r, 0)
		g_r_c_offset = self.start_offset_for(g_r, cell_cycle_num)
		return g_r_c_offset-g_r_0_offset-(g>0)*self.daughter_delay-(g==0)*(self.recovery_time-self.g1)

	def plot_cohort_translated(self, ax, g_r, cell_cycle_num):
		self.plot_cohort_offset(ax, g_r, cell_cycle_num, self.translated_offset(g_r, cell_cycle_num))

	def plot_gen(self, ax):


		for c in range(3):

			for g in range(2):
				for r in range(2):
					self.plot_cohort_translated(ax, (g, r), c)


		ax.set_xlim(-20, 300)
		ax.set_ylim(-2, 0.5)

		self.set_yticks(ax)
		ax.set_xlabel("Cell cycle time", fontsize=16)

		xticks = [-self.recovery_time+self.g1, -self.daughter_delay, 0, 
			self.lambd, self.lambd*2, self.lambd*3]

		for x in xticks: ax.axvline(x, c='gray', linestyle='dotted', lw=1, zorder=0)

		ax.set_xticks(xticks)
		ax.set_xticklabels(['$-\\mu_0$', '$-\\delta$', '0', '$\\lambda$',
		  '$2\\lambda$', '$3\\lambda$'], fontsize=14)


