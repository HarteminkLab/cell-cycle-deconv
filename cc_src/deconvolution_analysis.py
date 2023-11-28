
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DeconvolvedAnalysis:

    def __init__(self, model):


        f_path = 'output/xin-deconvolved_fs.csv'

        self.wt1_df = model.config.wt1_df
        self.wt2_df = model.config.wt2_df

        f = pd.read_csv(f_path)
        f.index = self.wt1_df.index
        self.f = f

        # --------------------

        # For analysis we will want the g values, f values, H, (TODO: gamma values), interval defintions
    
        # the g values live inside the model config.


        # --------------------

        # Have an ability to plot a single gene's profile (raw data, and deconvolved data), let's do this first


        # Then, we will calculate the peak to trough ratios for every gene

        return


    def plot_deconvolved_gene(self, gene_name, title=None):

        # If I want to plot the initial branch,
        # I need the branch name: i
        # the phases:   R, CG1, postG1
        # and their associated timepoints and indices:
            # Indices is done
            # timepoints are looked up in the intervals object

        from cc_src.sgd import get_orfname

        if gene_name in self.f.index.values:
            orfname = gene_name
        else:
            orfname = get_orfname(gene_name)

        # f_data = deconv_f.loc[orfname]
        # g_data = deconv_g.loc[orfname]
        # pred_g_data = self.predicted_g.loc[orfname]

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 3))
