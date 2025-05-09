
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

def save_figure_for_paper(save_path, dpi=350):
	plt.savefig(save_path, dpi=dpi, transparent=False, bbox_inches='tight')
	print(f"Saved figure to: {save_path}")
