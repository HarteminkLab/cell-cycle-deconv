
from matplotlib import pyplot as plt
import numpy as np


def plot_quantile_levels(deconvolved_gene_expression_t, cg1_genes):

    # from src.calcH_single_g1 import get_quantile_values
    # k = 3
    # gene_tx_means = deconvolved_gene_expression_t.loc[cg1_genes].mean(axis=1)
    # segments, qvals, lengths = get_quantile_values(
    #     gene_tx_means, np.linspace(0, 1, k+1)[1:-1])

    # fig = plt.figure(figsize=(24, 3.5))


    # def plt_early_vs_late_occ():

    # for i in range(k):
    #     plt.subplot(1, k, i+1)
    #     plt_early_vs_late_occ(segments[i], f"n={len(segments[i])}")
    # plt.suptitle(f"G1 peak genes, n={len(gene_tx_means)}", fontsize=18)
    # plt.subplots_adjust(top=0.8)
    # # Highly expressed in G2M seems different
    pass
