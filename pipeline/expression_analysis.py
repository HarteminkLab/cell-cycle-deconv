
# Let's analyze the first set of genes that have been deconvolved. Create a volcano plot
# get get an understanding of the mother daughter differences

from src.geneset import get_deconvolved_geneset
from glob import glob
from src.sgd import get_gene_name_orf_name
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_deconvolved_gene_expression(output_directory):
    genes = get_deconvolved_geneset()

    deconvolved_expression_filenames = glob(f"{output_directory}/genes_deconvolution/*.npy")

    gene_expression_Fs_list = []
    gene_names = []
    for i, filename in enumerate(deconvolved_expression_filenames):
        expression_F = np.load(filename)
        gene_name = filename.split('/')[-1].split('_')[0]
        orf_name, gene_name = get_gene_name_orf_name(gene_name)        
        gene_expression_Fs_list.append(expression_F)
        gene_names.append(orf_name)

    expression_Fs_df = pd.DataFrame(gene_expression_Fs_list, index=gene_names)

    return expression_Fs_df

def plot_volcano_cg1_dg1(expression_Fs_df, config1):

    from scipy.stats.distributions import norm

    cg1_dg1_avg_occ = (expression_Fs_df[config1.cg1_indices()].mean(axis=1) + 
        expression_Fs_df[config1.dg1_indices()].mean(axis=1))/2.

    cg1_dg1_max_ratio = np.log2(expression_Fs_df[config1.cg1_indices()].max(axis=1) / \
        expression_Fs_df[config1.dg1_indices()].max(axis=1))

    plot_data = pd.DataFrame({'avg_occ': cg1_dg1_avg_occ, 'max_ratio': cg1_dg1_max_ratio})

    plt.figure(figsize=(4, 3))
    plt.scatter(cg1_dg1_max_ratio + norm.rvs(0, 0.005, len(cg1_dg1_max_ratio)), 
        cg1_dg1_avg_occ, s=4)
    plt.xlabel("Ratio CG1/DG1")
    plt.xlim(-2, 2)
    plt.ylim(-1, 15)
    plt.xlabel("Ratio CG1/DG1")
    plt.ylabel("Mean expression level")
    plt.title("Mother/Daughter-specific gene expression")

    return plot_data

