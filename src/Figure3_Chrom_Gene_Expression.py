
import numpy as np
from src.global_config import GlobalConstants
from src.figure_configs import FiguresConfig
from matplotlib import pyplot as plt


class Figure3_Chrom_GeneExpression:
	"""docstring for Figure3_Chrom_GeneExpression"""

	def __init__(self, outdir):

		# Load deconvolved gene expression
		from src.gene_expression_deconv_analysis import GeneExpressionAnalysis

		gene_expression_dir = f'{outdir}/gene_expression/'
		self.gene_expression_a = GeneExpressionAnalysis(gene_expression_dir)
		self.outdir = outdir
		

	def partition_genes_phases(self):
		from src.config import load_configs_by_config_type

		config1, config2 = load_configs_by_config_type('shared')

		self.g1_indices = config1.get_Hpositions_for_phase('CG1')
		self.g2m_indices = config1.get_Hpositions_for_phase('G2M')
		self.s_indices = config1.get_Hpositions_for_phase('S')

		gene_expression = self.gene_expression_a.gene_expression_f

		g1_exp = gene_expression[self.g1_indices]
		g2m_exp = gene_expression[self.g2m_indices]
		s_exp = gene_expression[self.s_indices]

		max_g1_exp = g1_exp.max(axis=1)
		max_s_exp = s_exp.max(axis=1)
		max_g2m_exp = g2m_exp.max(axis=1)

		g1_genes = gene_expression.loc[(max_g1_exp > max_s_exp) & 
		                               (max_g1_exp > max_g2m_exp)].index
		g2m_genes = gene_expression.loc[(max_g2m_exp > max_s_exp) & 
		                                (max_g2m_exp > max_g1_exp)].index
		s_genes = gene_expression.loc[(max_s_exp > max_g2m_exp) & 
		                              (max_s_exp > max_g1_exp)].index

		self.g1_genes = g1_genes
		self.s_genes = s_genes
		self.g2m_genes = g2m_genes



	def load_ptrs(self):
		from src.peak_to_trough_anaysis import PeakToTroughAnalysis

		ptr_dir = f'{self.outdir}/chromatin/'
		ptr_analysis = PeakToTroughAnalysis(ptr_dir)
		self.ptr_analysis = ptr_analysis

	def correct_ptr_imgs(self):
		from src.geneset import get_deconvolved_geneset
		genes = get_deconvolved_geneset()
		genes['gene_index'] = np.arange(len(genes))

		ptr_imgs = self.ptr_analysis.ptr_imgs.reshape((-1, *GlobalConstants.IMAGE_SHAPE)).astype(float)
		ptr_imgs_strand_corrected = ptr_imgs.copy()
		crick_genes_indices = genes[genes['strand'] == '-'].gene_index.values
		ptr_imgs_strand_corrected[crick_genes_indices] = np.flip(ptr_imgs_strand_corrected[crick_genes_indices], axis=2)
		self.ptr_imgs_strand_corrected = ptr_imgs_strand_corrected
		self.genes = genes


	def select_k_genes(self, k=100):

		self.k = k
		gene_expression = self.gene_expression_a.gene_expression_f

		# G1, for the highest express G1 genes, what does the PTR chromatin look like?
		self.top_k_g1_genes = gene_expression.loc[self.g1_genes][self.g1_indices].median(axis=1).sort_values(ascending=False).head(k).index
		self.top_k_s_genes = gene_expression.loc[self.s_genes][self.s_indices].median(axis=1).sort_values(ascending=False).head(k).index
		self.top_k_g2m_genes = gene_expression.loc[self.g2m_genes][self.g2m_indices].median(axis=1).sort_values(ascending=False).head(k).index

		self.bottom_k_g1_genes = gene_expression.loc[self.g1_genes][self.g1_indices].median(axis=1).sort_values(ascending=False).tail(k).index
		self.bottom_k_s_genes = gene_expression.loc[self.s_genes][self.s_indices].median(axis=1).sort_values(ascending=False).tail(k).index
		self.bottom_k_g2m_genes = gene_expression.loc[self.g2m_genes][self.g2m_indices].median(axis=1).sort_values(ascending=False).tail(k).index

		self.top_g1_gene_indices = self.genes.loc[self.top_k_g1_genes].gene_index.values
		self.top_s_gene_indices = self.genes.loc[self.top_k_s_genes].gene_index.values
		self.top_g2m_gene_indices = self.genes.loc[self.top_k_g2m_genes].gene_index.values

		self.bottom_g1_gene_indices = self.genes.loc[self.bottom_k_g1_genes].gene_index.values
		self.bottom_s_gene_indices = self.genes.loc[self.bottom_k_s_genes].gene_index.values
		self.bottom_g2m_gene_indices = self.genes.loc[self.bottom_k_g2m_genes].gene_index.values

	def plot_gene_ptrs_phase(self):

		k = self.k

		plot_ptr_phases(self.ptr_imgs_strand_corrected, self.top_g1_gene_indices, self.top_s_gene_indices, 
                self.top_g2m_gene_indices,
               f"PTR {k} highest expressed genes")

		plot_ptr_phases(self.ptr_imgs_strand_corrected, self.bottom_g1_gene_indices, self.bottom_s_gene_indices, 
                self.bottom_g2m_gene_indices,
               f"PTR {k} lowest expressed genes")


def plot_ptr_phases(ptr_imgs, top_g1_gene_indices, 
                    top_s_gene_indices, top_g2m_gene_indices,
                   title):
    
    def plot_img(img, title):
        
        extent = [-GlobalConstants.PROM_LEN, GlobalConstants.GB_LEN,
            0, GlobalConstants.MAX_Y_LEN]
        plt.imshow(img, origin='lower', 
                              cmap='Spectral_r', vmin=1, vmax=3,
                  extent=extent, aspect='auto', interpolation='none')
        plt.title(title, fontsize=FiguresConfig.FIG_TITLE_FONTSIZE)
        plt.xticks([])
        plt.yticks([])
        plt.axvline(0, c='white', ls='solid', lw=0.75, alpha=0.5)
        
        plt.xticks(np.arange(-400, 600, 200))
    
    plt.figure(figsize=(13, 1.5))
    plt.subplot(1, 3, 1)
    plot_img(ptr_imgs[top_g1_gene_indices].mean(axis=0), "G1")

    plt.subplot(1, 3, 2)
    plot_img(ptr_imgs[top_s_gene_indices].mean(axis=0), "S")
    
    plt.subplot(1, 3, 3)
    plot_img(ptr_imgs[top_g2m_gene_indices].mean(axis=0), "G2/M")
    plt.colorbar()
    
    plt.suptitle(title, fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
    plt.subplots_adjust(top=0.6)