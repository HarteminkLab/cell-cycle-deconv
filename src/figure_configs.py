
from matplotlib import pyplot as plt


class FiguresConfig:

	FIG_SUPTITLE_FONTSIZE = 15
	FIG_TITLE_FONTSIZE = 13

	FIGSIZE_SHORT_WIDE = (9, 3)
	FIGSIZE_SHORT_EXTRAWIDE = (13, 3)
	FIGSIZE_SQUARE = (7, 7)
	FIGSIZE_SQUARE_WIDE = (7, 5)


def save_figure_for_paper(save_path):
	plt.savefig(save_path, dpi=350, transparent=True)
