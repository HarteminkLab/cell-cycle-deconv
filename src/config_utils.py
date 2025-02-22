
import numpy as np


def get_sample_indices(config, num_rows, phase):
    """
    Get indices to sample from G1 and postG1 phases to match row layout
    
    Parameters:
    -----------
    config : object
        Configuration object with methods:
        - get_Hpositions_for_phase(phase)
    num_g1_rows : int
        Number of rows to plot for G1 phase
    num_pg1_rows : int
        Number of rows to plot for postG1 phase
    g1_phase : str
        The G1 phase to sample from ('RG1', 'CG1', or 'DG1')
        
    Returns:
    --------
    g1_sampled : numpy.ndarray
        Sampled indices from G1 phase
    pg1_sampled : numpy.ndarray
        Sampled indices from postG1 phase
    """
    # Get indices for specific G1 phase and postG1
    indices = config.get_Hpositions_for_phase(phase)
    
    # Sample evenly from each phase
    indices_sampled = np.linspace(0, len(indices)-1, num_rows, dtype=int)
    indices_sampled = indices[indices_sampled]
        
    return indices_sampled
