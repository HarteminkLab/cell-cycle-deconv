


expression_mean = gene_clustering.gene_expression[t_indices].mean(axis=1)
pas_gb_nuc_occ = all_pas_gb_nuc_occ[t_indices].mean(axis=1)

compare_acetyl_tx = genes_h3k56ac.copy()
compare_acetyl_tx['mean_expression'] = expression_mean
compare_acetyl_tx['mean_pas_gene_body_occupancy'] = pas_gb_nuc_occ
compare_acetyl_tx['mean_entropy'] = mean_entropy

plt.figure(figsize=(4, 3))

plt.scatter(compare_acetyl_tx.mean_pas_gene_body_occupancy, 
            -compare_acetyl_tx.mean_entropy, s=1,
           c=compare_acetyl_tx.mean_acetylation, alpha=1,
           vmin=-1, vmax=1, cmap='RdBu_r')
plt.ylabel("Average organization")
plt.xlabel("Average PAS occupancy")
cbar = plt.colorbar()
cbar.ax.set_ylabel("H3K56 acetylation", rotation=270, va='bottom', labelpad=0)
cbar.ax.set_yticks([-1, 0, 1])
plt.suptitle(f"Occupancy vs organization, n={len(compare_acetyl_tx)}")

# Shows a potential characterization of nucleosomes:
# Acetylation provides evidence for greater nucleosome organization given
# the same nucleosome occupancy
# 
# Can we distill this further? Does this help with the buffering story or is 
# something different?
#
# Will we be able to look at different nucleosomes with the same occupancy
# and correlate different organizaiton scores to acetylation?
# Does this change over time? What does the deconvolution algorithm do to help with this?

