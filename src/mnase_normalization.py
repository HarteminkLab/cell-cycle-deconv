

import numpy


def normalize_to_target_distribution(data, target_distribution):
    """
    Normalize the data array to match the target fragment length distribution.
    """

    current_distributions = data.sum(axis=2)  # shape: (16, 251)
    
    epsilon = 1e-10
    scaling_factors = target_distribution / ((current_distributions[:, :, None]) + epsilon)
    
    # Apply scaling factors to the data
    normalized_data = data * scaling_factors
    
    # Mean normalize to 1.0
    return normalized_data / normalized_data.mean()

def normalize_to_target_sums(data, target_sums):
    """
    Normalize the data array to match target sum for each timepoint.
    """
    # Calculate current sum for each timepoint
    current_sums = data.sum(axis=(1, 2))  # shape: (16,)
    
    # Calculate scaling factor for each timepoint
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    scaling_factors = target_sums / (current_sums + epsilon)  # shape: (16,)
    
    # Reshape for broadcasting
    scaling_factors = scaling_factors.reshape(16, 1, 1)
    
    # Apply scaling to maintain the relative proportions within each timepoint
    # while scaling to the target sum
    normalized_data = data * scaling_factors
    
    normalized_data = normalized_data / normalized_data.mean() * target_sums.mean()
        
    return normalized_data 

