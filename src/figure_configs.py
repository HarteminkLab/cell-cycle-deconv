
from matplotlib import pyplot as plt


class FiguresConfig:

	FIG_SUPTITLE_FONTSIZE = 15
	FIG_TITLE_FONTSIZE = 14
	FIG_LABEL_FONTSIZE = 13

	FIGSIZE_SHORT = (6, 4)
	FIGSIZE_SHORT_WIDE = (9, 4)
	FIGSIZE_SHORT_EXTRAWIDE = (13, 4)
	FIGSIZE_SQUARE = (7, 7)
	FIGSIZE_SQUARE_WIDE = (6.25, 5)

def save_figure_for_analysis(save_path):
	plt.savefig(save_path, dpi=200, transparent=False)

def save_figure_for_paper(save_path):
	plt.savefig(save_path, dpi=350, transparent=True)
