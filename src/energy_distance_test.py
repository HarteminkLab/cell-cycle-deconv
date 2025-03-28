import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt

def energy_distance(data1, data2, metric='euclidean'):
    """
    Calculate the energy distance between two samples.
    
    The energy distance is a measure of statistical distance between probability
    distributions and is defined as:
    
    E(F, G) = 2E||X - Y|| - E||X - X'|| - E||Y - Y'||
    
    where X, X' are independent random vectors from distribution F, and
    Y, Y' are independent random vectors from distribution G.
    
    Parameters:
    -----------
    data1, data2 : numpy.ndarray
        Input data arrays, shape (n_samples, n_features)
    metric : str, optional
        Distance metric to use. Default is 'euclidean'.
        
    Returns:
    --------
    energy_dist : float
        The energy distance between the two samples.
    """
    n1, n2 = len(data1), len(data2)
    
    # Calculate pairwise distances within and between samples
    # Between samples
    dist_between = cdist(data1, data2, metric=metric)
    mean_dist_between = np.mean(dist_between)
    
    # Within first sample
    if n1 > 1:
        dist_within1 = pdist(data1, metric=metric)
        mean_dist_within1 = np.mean(dist_within1)
    else:
        mean_dist_within1 = 0
    
    # Within second sample
    if n2 > 1:
        dist_within2 = pdist(data2, metric=metric)
        mean_dist_within2 = np.mean(dist_within2)
    else:
        mean_dist_within2 = 0
    
    # Calculate energy distance
    energy_dist = 2 * mean_dist_between - mean_dist_within1 - mean_dist_within2
    
    return energy_dist

def energy_distance_permutation_test(data1, data2, n_permutations=999, 
                                     metric='euclidean', plot_hist=False):
    """
    Perform a permutation test based on energy distance.
    
    Parameters:
    -----------
    data1, data2 : numpy.ndarray
        Input data arrays, shape (n_samples, n_features)
    n_permutations : int, optional
        Number of permutations to perform. Default is 999.
    metric : str, optional
        Distance metric to use. Default is 'euclidean'.
    plot_hist : bool, optional
        Whether to plot a histogram of permutation statistics. Default is False.
        
    Returns:
    --------
    dict
        Dictionary containing test results and diagnostics
    """
    np.random.seed(123)
    n1, n2 = len(data1), len(data2)
    combined = np.vstack([data1, data2])
    
    # Calculate observed energy distance
    observed_energy = energy_distance(data1, data2, metric=metric)
    
    # Perform permutation test
    perm_energies = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Randomly shuffle labels
        indices = np.random.permutation(n1 + n2)
        perm_data1 = combined[indices[:n1]]
        perm_data2 = combined[indices[n1:]]
        
        # Calculate energy distance for permuted data
        perm_energies[i] = energy_distance(perm_data1, perm_data2, metric=metric)
    
    # Calculate p-value (proportion of permutation statistics >= observed)
    p_value = np.mean(perm_energies >= observed_energy)
    
    # Also calculate the proportion of permutation statistics <= observed
    # for diagnostic purposes
    p_value_less = np.mean(perm_energies <= observed_energy)
    
    # Print results
    print(f"Energy Distance Test Results:")
    print(f"  Observed energy distance: {observed_energy:.6f}")
    print(f"  Mean of permuted energy distances: {np.mean(perm_energies):.6f}")
    print(f"  p-value (H1: distributions differ): {p_value:.6f}")
    
    # Interpretation guidance
    if observed_energy > np.mean(perm_energies):
        print("\nInterpretation:")
        print("  The observed energy distance is GREATER than expected by chance.")
        print("  This suggests the two groups are MORE DIFFERENT than random expectation.")
    else:
        print("\nInterpretation:")
        print("  The observed energy distance is LESS than expected by chance.")
        print("  This suggests the two groups are MORE SIMILAR than random expectation.")
    
    # Plot histogram if requested
    if plot_hist:
        plt.figure(figsize=(10, 6))
        plt.hist(perm_energies, bins=30, alpha=0.7, label='Permutation Statistics')
        plt.axvline(observed_energy, color='red', linestyle='dashed', 
                   linewidth=2, label=f'Observed Energy Distance: {observed_energy:.6f}')
        plt.axvline(np.mean(perm_energies), color='blue', linestyle='dashed', 
                   linewidth=2, label=f'Mean of Permutations: {np.mean(perm_energies):.6f}')
        plt.title('Energy Distance Permutation Test')
        plt.xlabel('Energy Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    return {
        'observed_stat': observed_energy,
        'p_value': p_value,
#         'p_value_less': p_value_less,
#         'perm_stats_mean': np.mean(perm_energies),
#         'perm_stats_std': np.std(perm_energies),
#         'perm_stats': perm_energies
    }
