
import numpy as np


def find_nearest_index(tps, timepoint):
    """Find the nearest index in the time points array for the given timepoint."""
    return np.abs(tps - timepoint).argmin()

def handle_wrap_around(timepoint, min_tps, max_tps, phase_duration):
    """Handle wrap-around logic for time points."""
    if timepoint < min_tps:
        timepoint += phase_duration
    elif timepoint > max_tps:
        timepoint -= phase_duration
    return timepoint

def get_delta_rep_index(g1_indices, postg1_indices, g1_tps, postg1_tps, index_of_rep_H, delta_in_min):
    """
    Return the indices and timepoints of the index of the span around the replication time.

    Parameters:
    index_of_rep_H (int): Index in H which needs to be converted into mother-only indices.
    delta_in_min (float): The delta value in minutes for computing the span around the replication time.

    Returns:
    tuple: (index_minus_delta_in_H, index_plus_delta_in_H, timepoint_minus_delta, timepoint_plus_delta)
    """
    # Determine if the index is in g1 or postg1
    if index_of_rep_H in g1_indices:
        phase_tps = g1_tps
        phase_indices = g1_indices
        other_phase_tps = postg1_tps
        other_phase_indices = postg1_indices
    elif index_of_rep_H in postg1_indices:
        phase_tps = postg1_tps
        phase_indices = postg1_indices
        other_phase_tps = g1_tps
        other_phase_indices = g1_indices
    else:
        raise ValueError("Index is not in the g1 or postg1 phase")

    # Find the corresponding time point for the index
    index_in_phase = np.where(phase_indices == index_of_rep_H)[0][0]
    timepoint_of_rep = phase_tps[index_in_phase]

    # Calculate the new time points
    timepoint_minus_delta = timepoint_of_rep - delta_in_min
    timepoint_plus_delta = timepoint_of_rep + delta_in_min

    # Phase duration for wrap-around handling
    phase_duration = other_phase_tps[-1] - phase_tps[0] + (other_phase_tps[1] - other_phase_tps[0])

    # Handle wrap-around for time points
    timepoint_minus_delta = handle_wrap_around(timepoint_minus_delta, phase_tps[0], other_phase_tps[-1], phase_duration)
    timepoint_plus_delta = handle_wrap_around(timepoint_plus_delta, phase_tps[0], other_phase_tps[-1], phase_duration)

    # Find the nearest indices in the phase time points
    def get_index_and_timepoint(timepoint, phase_tps, phase_indices, other_phase_tps, other_phase_indices):
        if timepoint >= phase_tps[0] and timepoint <= phase_tps[-1]:
            index_in_phase = find_nearest_index(phase_tps, timepoint)
            index_in_H = phase_indices[index_in_phase]
            timepoint = phase_tps[index_in_phase]
        else:
            index_in_phase = find_nearest_index(other_phase_tps, timepoint)
            index_in_H = other_phase_indices[index_in_phase]
            timepoint = other_phase_tps[index_in_phase]
        return index_in_H, timepoint

    index_minus_delta_in_H, timepoint_minus_delta = get_index_and_timepoint(
        timepoint_minus_delta, phase_tps, phase_indices, other_phase_tps, other_phase_indices)
    
    index_plus_delta_in_H, timepoint_plus_delta = get_index_and_timepoint(
        timepoint_plus_delta, phase_tps, phase_indices, other_phase_tps, other_phase_indices)

    return (index_minus_delta_in_H, index_plus_delta_in_H, timepoint_minus_delta, timepoint_plus_delta)