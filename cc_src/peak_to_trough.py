
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_combined_mother_daughter_ptr(mother_ptrs, daughter_ptrs):
    """
    Calculate the combined mother and daughter PTR as the geometric mean of combining both PTR
    values: 3rd root of [   ptr_mother^2     *     daughter_ptr ]
    """
    combined_ptr = np.power(mother_ptrs**2 * daughter_ptrs, 1/3.)
    return combined_ptr


def calculate_ptr(data):
    """
    Calculate the peak to trough ratio of a data set, where each row is a gene
    and each column is a time point. 

    The 80% expression for a gene is computed as the peak, 20% for the trough.

    The ratio is the (P+1)/(T+1), +1 to handle divide by zero (TODO: Maybe adjust pseudocount?)
    """

    # If two dimensions, do it row-wise
    if data.ndim == 2:
        q80 = np.quantile(data, q=0.8, axis=1)
        q20 = np.quantile(data, q=0.2, axis=1)

    # otherwise one dimension
    else:
        q80 = np.quantile(data, q=0.8)
        q20 = np.quantile(data, q=0.2)

    ptr = (q80+1) / (q20+1)
    return ptr


def load_replicate2_gene_expression_ptr():
    times = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140]).astype(str)
    times_int = times.astype(int)
    TPM_df = pd.read_csv('data/gene_TPM.csv').set_index('orf_name')

    isna = TPM_df['gene'].isna()
    TPM_df.loc[isna, 'gene'] = TPM_df.index[isna]
    TPM_df = TPM_df.set_index('gene')

    tpm_log2 = TPM_df.copy()
    tpm_log2[times] = np.log2(TPM_df[times]+1)
    tpm_log2.head(1)

    TPM_df['ptr'] = calculate_ptr(TPM_df[times].values)
    tpm_log2['ptr'] = calculate_ptr(tpm_log2[times].values)

    return TPM_df, tpm_log2


def load_xin_gene_expression_ptr():
    """
    Load the Xin Guo paper's raw gene expression data.
    """

    # Load the ORF names, as the gene expression data is unnamed rows, this will
    # allow us to know what gene belongs to which row. This game from the gene.lst file
    # in the xin deconvolution code base
    xin_gene_names = pd.read_csv('data/xin_data/raw/orf_names.txt', sep='\t', names=['genename'])

    # Load the gene expression data, get the column numbers so we can place them in header indices
    # get the number of columns for loading
    #
    # TODO: get the actual time points for this data for the headers
    #
    data_path = 'data/xin_data/raw/replicate2_gene_expression.txt'
    dummy_4cols = pd.read_csv(data_path, sep='\t')
    numcols = len(dummy_4cols.columns)
    xin_g = pd.read_csv(data_path, sep='\t', names=np.arange(numcols))
    xin_g['genename'] = xin_gene_names

    # Compute the peak-to-trough ratio
    xin_g = xin_g.set_index('genename')
    xin_g['ptr'] = calculate_ptr(xin_g.values)

    # Compute the peak-to-trough ratio for the log2 gene expression
    xin_deconvolved_log2 = np.log2(xin_g+1)
    xin_deconvolved_log2['ptr'] = calculate_ptr(xin_deconvolved_log2.values)

    return xin_g, xin_deconvolved_log2


def load_xin_deconvolved_gene_expression_ptr():
    """
    Load the Xin Guo paper's deconvolved gene expression data
    """

    xin_gene_names = pd.read_csv('data/xin_data/deconvolved/allgenes.gm', 
                                 sep='\t', names=['gene', 'orf_name', '?'])

    xin_deconvolved_f = pd.read_csv('data/xin_data/deconvolved/allgenes_from_deconv2_allgenes.f', 
                                    sep='\t', names=np.arange(258))
    xin_deconvolved_f['orf_name'] = xin_gene_names['orf_name']
    xin_deconvolved_f = xin_deconvolved_f.set_index('orf_name')

    # # Trying a different deconvolve dataset
    # d = pd.read_csv('data/xin_data/deconvolved/deconvolved_profiles.tsv',
    #                                  sep='\t')
    # d = d.set_index('SystematicName')[d.columns[3:]].iloc[1:]
    # xin_deconvolved_f = d

    # Calculate PTR for deconvolved data
    xin_deconvolved_f['ptr'] = calculate_ptr(xin_deconvolved_f.values)

    xin_deconvolved_f_log2 = np.log2(xin_deconvolved_f+1)
    xin_deconvolved_f_log2['ptr'] = calculate_ptr(xin_deconvolved_f_log2.values)

    return xin_deconvolved_f, xin_deconvolved_f_log2


def plot_peak_to_trough_scatter(xin_deconvolvedlog, xin_rawlog, suffixes=[], title=None, genes_to_plot=[], genes=[]):
    keys = 'ptr' + suffixes[0], 'ptr' + suffixes[1]

    plt.figure(figsize=(8, 8))
    plt_data = xin_deconvolvedlog[['ptr']].join(xin_rawlog[['ptr']], lsuffix=suffixes[0], rsuffix=suffixes[1])
    plt.scatter(plt_data[keys[0]], plt_data[keys[1]], s=10)
    plt.title(title)
    plt.xlabel(suffixes[0].replace('_', ' ').replace("ptr", "PTR"))
    plt.ylabel(suffixes[1].replace('_', ' ').replace("ptr", "PTR"))

    gene_rows_to_plot = genes[genes['gene'].isin(genes_to_plot)]

    selected_plt_data = plt_data.loc[gene_rows_to_plot.index]
    plt.scatter(selected_plt_data[keys[0]], selected_plt_data[keys[1]], s=40, c='red')
    
    for idx, row in selected_plt_data.join(genes[['gene']]).iterrows():
        plt.text(row[keys[0]], row[keys[1]], row['gene'], fontsize=30)


def plot_genename(genename, expression_df1, expression_df2):
    from src.sgd import get_orfname

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    orf_name = get_orfname(genename)
    indices = expression_df1.columns[:-1]
    data = expression_df1.loc[orf_name][indices]
    ptr = calculate_ptr(data)
    ax1.plot(data, lw=3)
    ax1.set_title(f"Original, PTR={ptr:.2f}")
    ax1.set_yscale('log')

    indices = expression_df2.columns[:-1]
    data = expression_df2.loc[orf_name][indices]
    ptr = calculate_ptr(data)
    ax2.plot(np.arange(len(indices)),
             expression_df2.loc[orf_name][indices], c='orange', lw=3)
    
    ax2.set_title(f"Deconvolved, PTR={ptr:.2f}")
    ax2.set_yscale('log')

    plt.suptitle(genename)
    