
from matplotlib import pyplot as plt


class FiguresConfig:

	FIG_SUPTITLE_FONTSIZE = 16
	FIG_TITLE_FONTSIZE = 13
	FIG_LABEL_FONTSIZE = 12

	FIGSIZE_SHORT = (6, 4)
	FIGSIZE_SHORT_WIDE = (9, 4)
	FIGSIZE_SHORT_EXTRAWIDE = (13, 4)
	FIGSIZE_SQUARE = (6, 6)
	FIGSIZE_SQUARE_WIDE = (6.25, 5)

	FIGSIZE_WIDE = (10, 6)

def save_figure_for_analysis(save_path):
	plt.savefig(save_path, dpi=200, transparent=False, bbox_inches='tight')
	print(f"Saved figure to: {save_path}")

def save_figure_for_paper(save_path, fig=None, dpi=350):
	sav_obj = plt if fig is None else fig
	sav_obj.savefig(save_path, dpi=dpi, transparent=False, bbox_inches='tight')
	print(f"Saved figure to: {save_path}")

tf_colors = {
	
	# Pioneer TFs
	'Abf1': '#505050',    # dark gray (pioneer)
	'Rap1': '#9b9b9b',    # medium gray (pioneer)
	'Reb1': '#a8a8a8',    # lighter gray (pioneer)

	# General/regulatory TFs
	'Spt15': 'steelblue',   # red (existing)
	'Cin5': '#1abc9c',    # teal (existing)

	# Cell-cycle specific TFs
	'Fkh1': '#e74c3c',    # pink/magenta (existing)
	'Fkh2': '#ec407a',    # lighter pink (similar to Fkh1)
	'Mcm1': '#2c9645',    # green (existing)
	'Mbp1': '#3498db',    # blue (existing)
	
	# General chromatin organizer
	'Tbf1': '#8b7355',    # muted brown

	'Sum1': 'yellow',
	'Nhp6A': 'orange',
	
	'other': '#555555'    # gray (existing)
}