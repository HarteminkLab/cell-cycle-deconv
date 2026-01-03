
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import collections as mc
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects


class ORFAnnotationPlotter:
	"""
	Class to plot the gene annotations from SGD.
	"""

	def __init__(self, orfs, origins, aux_annotations):

		self.orfs = orfs
		self.origins = origins
		self.aux_annotations = aux_annotations

		self.show_spines = True
		self.show_minor_ticks = False
		self.plot_tick_labels = True
		self.span = None
		self.chrom = None
		self.inset = (0, 10.0)
		self.triangle_width = 10
		self.text_horizontal_offset = 10
		self.plot_orf_names = True
		self.height = 40
		self.text_vertical_offset = 18
		self.y_padding = 10

	def set_chrom_span(self, chrom, span):

		self.chrom = chrom
		self.span = int(span[0]), int(span[1])

		span_width = self.span[1] - self.span[0]
		self.span_width = span_width

		# triangle is proportion of the span width
		self.triangle_width = span_width/400.*2.5

		# overlap triangle with rect
		self.epsilon = span_width * 1e-4

		self.text_horizontal_offset = span_width * 1e-2

	def get_color(self, watson, gene_type):
		colors = {
			"Verified": cm.Blues(0.8),
			"Putative": cm.Blues(0.5),
			"Dubious": "#898888"
		}
		if not watson:
			colors['Verified'] = cm.Reds(0.5)
			colors['Putative'] = cm.Reds(0.3)

		colors['Uncharacterized'] = colors['Putative']
		color = colors[gene_type]
		return color

	def plot_orf_annotation(self, ax, start, end, name, CDS_introns=None, 
		watson=True, offset=False, gene_type="Verified", TSS=np.nan, PAS=np.nan,
		plot_orf_names=True, text_horizontal_offset=None, y=None, color=None, 
		text_style='italic', fontsize=12, flip_genome=False):

		if color is None:
			color = self.get_color(watson, gene_type)

		rect_width = end - start - self.triangle_width

		if text_horizontal_offset is None:
			text_horizontal_offset = self.text_horizontal_offset

		inset = self.inset

		if watson:
			rect_start = start
			# add one to overlap triangle with rect
			triangle_start = end - self.triangle_width - self.epsilon*2 
			y_baseline = offset * (self.y_padding + self.height)
			text_start = start + text_horizontal_offset
		else:
			# add one to overlap triangle with rect
			triangle_start = start + self.epsilon 
			rect_start = start + self.triangle_width
			y_baseline = (offset+1) * (-self.y_padding-self.height)
			text_start = end - text_horizontal_offset

		if flip_genome:
			if watson:
				text_start = start + text_horizontal_offset
				y_baseline = (offset+1) * (-self.y_padding-self.height)
			else:
				text_start = end - text_horizontal_offset
				y_baseline = offset * (self.y_padding + self.height)

		if y is not None:
			y_baseline = y

		# plot pointed rectangle for ORF
		plot_rect(ax, rect_start, y_baseline, rect_width, self.height, color, 
			inset=inset)

		plot_iso_triangle(ax, triangle_start, y_baseline, 
			self.triangle_width, self.height, color, facing_right=watson,
			inset=inset[1])

		# plot introns as rectangles overtop of ORF
		if CDS_introns is not None and len(CDS_introns) > 0:
			for idx, intron in CDS_introns.iterrows():
				if intron['cat'] == 'intron':
					intron_width = intron.stop - intron.start
					# plot introns as a lighter version of the CDS
					plot_rect(ax, intron.start, y_baseline, intron_width, 
						self.height, 'white', inset=inset)
					plot_rect(ax, intron.start, y_baseline, intron_width, 
						self.height, color, fill_alpha=0.25, inset=inset)

		# plot name of ORF
		if plot_orf_names:

			# if genome is flipped, adjust ha of text
			ha = 'left' if (not flip_genome and watson) or \
						   (flip_genome and not watson) else 'right'

			plot_text(ax,
				text_start, 
				y_baseline+self.text_vertical_offset, 
				name, color, ha=ha,
				fontsize=fontsize,
				text_style=text_style)

		# Plot TSS PAS line indicators
		if TSS != np.nan or PAS != np.nan:
			plot_TSS_PAS(ax, start, end, TSS, PAS, 
				y_baseline, self.height, color, flipped=watson, inset=inset[1])

		draw_TSS_arrow(ax, TSS, y_baseline+inset[1], flip=not watson, 
			plot_width=self.triangle_width/2., color=color)

	def plot_auxiliary(self, ax):
		from src.plot_helpers import plot_rect2

		span = self.span
		chrom = int(self.chrom)
		aux_annotations = self.aux_annotations

		aux_annotations_to_plot = aux_annotations[(aux_annotations['chr'] == chrom) & 
				 ((aux_annotations['start'] >= span[0]) & 
				  (aux_annotations['stop'] <= span[1]))]

		for _, aux in aux_annotations_to_plot.iterrows():

			x1, x2 = aux.start, aux.stop

			height = 30
			h_2 = height//2
			offset = 20
			if aux.strand == '+':
				y1, y2 = -h_2+offset, h_2+offset
			else:
				y1, y2 = -h_2-offset, h_2-offset

			plot_rect2(ax, x1, y1, x2, y2, facecolor='#ba9b8d', zorder=0)

			text = ax.text((x1+x2)/2, (y1+y2)/2., aux['name'], 
				rotation=0, color='white', ha='center', fontsize=5,
				va='center', clip_on=True, zorder=65)
			text.set_path_effects([path_effects.Stroke(linewidth=1.5, 
				foreground='#ba9b8d'), path_effects.Normal()])

	def plot_origins(self, ax):

		from src.plot_helpers import plot_rect2

		span = self.span
		chrom = int(self.chrom)
		origins = self.origins

		origins_to_plot = origins[(origins['chr'] == chrom) & 
				 ((origins['start'] >= span[0]) & 
				  (origins['stop'] <= span[1]))]

		for _, origin_sgd in origins_to_plot.iterrows():

			x1, x2 = origin_sgd.start, origin_sgd.stop
			y1, y2 = -80, 80
			plot_rect2(ax, x1, y1, x2, y2, facecolor='gray', zorder=0)

			text = ax.text((x1+x2)/2, 0, origin_sgd.ars_name, 
				rotation=90, color='white', ha='center', fontsize=7,
				va='center', clip_on=True, zorder=65)
			text.set_path_effects([path_effects.Stroke(linewidth=1.5, 
				foreground='gray'), path_effects.Normal()])


	def plot_orf_annotations(self, ax, 
		orf_classes=['Verified', 'Uncharacterized', 'Dubious'],
		custom_orfs=None, should_auto_offset=True, flip_genome=False):

		if custom_orfs is None:
			orfs = self.orfs
		else: orfs = custom_orfs

		span = self.span
		chrom = int(self.chrom)

		from src.sgd import select_genes_in_window

		# Find genes within this span, pad to handle TSS and PASs
		genes = select_genes_in_window(orfs, chrom, span, orf_classes)

		try:
			genes = genes.sort_values(['strand', 'start']).reset_index()

		# older pandas version
		except AttributeError:
			genes = genes.sort(['strand', 'start']).reset_index()

		tick_intervals = 500, 100

		for idx, gene in genes.iterrows():

			gene_name = gene['gene']
			name = gene['orf_name']

			custom_lookup_genes = {
				'YLR194C': 'NCW2',
				'YMR178W': 'FPY1',
			}

			# Special case with updated gene name from sacCer3
			if gene.orf_name in custom_lookup_genes:
				gene = gene.copy()
				gene_name = custom_lookup_genes[gene.orf_name]
				name = gene.orf_name
				gene.classification = 'Verified'

			if self.plot_orf_names:
				if gene_name is not None and not name == gene_name:
					name = "{} / {}".format(gene_name, gene['orf_name'])
			else:
				name = gene_name

			gene_introns = None

			offset = False
			if should_auto_offset: offset = (idx % 2) == 0

			self.plot_orf_annotation(ax, gene.start,
					  gene.stop, name,
					  CDS_introns=gene_introns,
					  watson=(gene.strand == "+"),
					  gene_type=gene.classification,
					  offset=offset,
					  TSS=gene['TSS'],
					  PAS=gene['PAS'],
					  flip_genome=flip_genome)

		ax.set_xticks([])

		if flip_genome:
			ax.set_xlim(span[1], span[0])
		else:
			ax.set_xlim(*span)

		self.plot_origins(ax)
		self.plot_auxiliary(ax)

		ax.set_yticks([])
		ax.set_ylim(-110, 110)

		return ax

	def plot_centromere(self, ax, start, end, name,
			watson=True, offset=False,
			text_horizontal_offset=None, y=None):

		color='#4A4A4A'

		rect_width = end - start - self.triangle_width

		if text_horizontal_offset is None:
			text_horizontal_offset = self.text_horizontal_offset

		inset = self.inset

		if watson:
			rect_start = start
			# add one to overlap triangle with rect
			triangle_start = end - self.triangle_width - self.epsilon*2 

			y_baseline = offset * (self.y_padding + self.height)
			text_start = start
			ha = 'left'
		else:
			# add one to overlap triangle with rect
			triangle_start = start + self.epsilon 
			rect_start = start + self.triangle_width
			y_baseline = (offset+1) * (-self.y_padding-self.height)
			text_start = end
			ha = 'right'

		if y is not None:
			y_baseline = y

		# plot pointed rectangle
		plot_rect(ax, rect_start, y_baseline, rect_width, self.height, color, 
			inset=inset)

		plot_iso_triangle(ax, triangle_start, y_baseline, 
			self.triangle_width, self.height, color, facing_right=watson,
			inset=inset[1])

		text = ax.text(text_start, y_baseline+self.text_vertical_offset,
			name, fontsize=12, clip_on=True, zorder=65, 
					   rotation=0, va='center', ha=ha,
					   fontdict={'fontname': 'Open Sans'},
					   color='white')
		text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground=color),
							   path_effects.Normal()])


def plot_TSS_PAS(ax, start, end, TSS, PAS, y, height, color, flipped=False, 
	inset=0):
	"""# Plot TSS PAS lines, if TSS or PAS does not exist use ORF start
		# and end boundaries"""

	y_bottom = y+inset/2.
	y_top = y+height-inset/2


	if TSS is not None:
		ax.plot([TSS, TSS], [y_bottom, y_top], color=color, lw=1.5, zorder=111)

	if PAS is not None:
		ax.plot([PAS, PAS], [y_bottom, y_top], color=color, lw=1.5, zorder=111)

	if not flipped:
		if np.isnan(TSS): TSS = start
		if np.isnan(PAS): PAS = end
	else:
		if np.isnan(TSS): TSS = end
		if np.isnan(PAS): PAS = start

	TSS_span = TSS, PAS    
	ax.plot(TSS_span, [y+height/2., y+height/2.], lw=1.5, color=color, zorder=60)


def draw_TSS_arrow(ax, x, y, flip, plot_width, color):

	tri_width, tri_height = plot_width*5, 12
	tss_height = 32
	tss_width = plot_width*6
	lw=1.5

	if flip: 
		tss_width = -tss_width
		tri_width = -tri_width

	plot_iso_triangle(ax, x+tss_width, y+tss_height-tri_height/2, 
		tri_width, tri_height, color)
	ax.plot([x, x+tss_width], [y+tss_height, y+tss_height], c=color, lw=lw)
	ax.plot([x, x], [y, y+tss_height], c=color, lw=lw)


def plot_iso_triangle(ax, x, y, width, height, color, facing_right=True,
	inset=0):
	"""
	Plot left or right pointing isosceles triangle for end cap of ORF
	"""

	y_bottom = y+inset/2.
	y_top = y+height-inset/2.
	x_flat = x

	if facing_right: 
		x_pointed = x+width
	else: 
		x_pointed = x
		x_flat += width

	X = np.array([[x_flat,y_bottom], [x_flat,y_top], 
		[x_pointed, (y_bottom+y_top)/2.0]])

	triangle = plt.Polygon(X, color=color, linewidth=0, zorder=2)
	ax.add_patch(triangle)

def plot_text(ax, start, y, name, color, flipped=False, text_style='italic',
			  fontsize=22, rotation='horizontal', ha=None):

	if ha is None:
		ha = 'left'
		if flipped: ha = 'right'

	text = ax.text(start, y, name, fontsize=fontsize, clip_on=True, zorder=65, 
				   rotation=rotation, va='center', ha=ha, 
				   fontdict={'fontname': 'Open Sans', 'fontweight': 'regular',
				   'style': text_style},
				   color='white')
	text.set_path_effects([path_effects.Stroke(linewidth=3, foreground=color),
						   path_effects.Normal()])


def plot_rect(ax, x, y, width, height, color=None, facecolor=None, 
	edgecolor=None, ls='solid', fill_alpha=1., zorder=40, lw=0.0, 
	inset=(0.0, 0.0), fill=True, joinstyle='round'):
	"""
	Plot a rectangle for ORF plotting
	"""

	if edgecolor is None: edgecolor = color
	if facecolor is None: facecolor = color

	patch = ax.add_patch(
					patches.Rectangle(
						(x, y + inset[1]/2.0),   # (x,y)
						width, height - inset[1], # size
						facecolor=facecolor,
						edgecolor=edgecolor,
						lw=lw,
						joinstyle=joinstyle,
						ls=ls,
						fill=fill,
						alpha=fill_alpha,
						zorder=zorder
					))


def plot_gene_annotation(ax, start, end, y_baseline, height, color, 
						 inset, triangle_start, triangle_width, TSS, PAS, watson):

	rect_start = start
	triangle_start = end-triangle_width
	rect_width = end - start - triangle_width

	# Plot pointed rectangle for ORF
	plot_rect(ax, rect_start, y_baseline, rect_width, height, color, inset=inset)
	
	# Plot isosceles triangle
	plot_iso_triangle(ax, triangle_start, y_baseline, 
					  triangle_width, height, color, facing_right=watson,
					  inset=inset[1])
	
	# Draw TSS arrow
	draw_TSS_arrow(ax, TSS, y_baseline+inset[1]+height, flip=not watson, 
				   plot_width=3, color=color)

	# plot_TSS_PAS(ax, start, end, TSS, PAS, 
	# 			y_baseline, height, color, flipped=watson, inset=inset[1])


def load_default_orf_plotter(output_directory=None):

	from src.sgd import read_sgd_genes, read_nondubious_genes_dataset, \
		load_origins_sgd, load_aux_annotations, read_geneset_with_computed_regions
	from src.orf_plotter import ORFAnnotationPlotter
	from src.orf_plotter import load_default_orf_plotter
	from src.transcripts_dataset import load_transcripts_sets

	if output_directory is None:
		nondub_genes = read_geneset_with_computed_regions()
	else:
		nondub_genes, _ = load_transcripts_sets(output_directory)

	aux_annotations = load_aux_annotations()
	origins = load_origins_sgd()
	# All genes, dubious genes may be relevant
	all_genes = read_sgd_genes(remove_chr_roman=True)

	# Nondubious genes, these have the proper columns for orf plotting
	all_genes = all_genes.join(nondub_genes[['TSS', 'PAS']])

	orf_plotter = ORFAnnotationPlotter(all_genes, origins, aux_annotations)
	return orf_plotter
