
from matplotlib import pyplot as plt
import numpy as np


def plot_phase_stack(tp, prev_vec, cur_vec, name):
    from src.model import color_for_key

    if prev_vec is None:
        prev_vec = np.zeros_like(cur_vec)
    
    cur_stack = prev_vec+cur_vec
    
    plt.fill_between(tp, prev_vec, cur_stack, color=color_for_key(name), label=name)

    return cur_stack

    
def plot_H_as_growth_curve(model):

    plt.figure(figsize=(8, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(model.H, aspect='auto', vmax=0.02, cmap='viridis')
    from cc_src.plot_helpers import plot_H_as_growth_curve

    plt.subplot(1, 2, 2)
    tp = model.config.WT1_TIMEPOINTS
    H = model.H[:len(tp)]

    phase_columns = model.config.phase_columns
    keys = list(phase_columns.keys())

    plt.ylim(0, 2.)

    phases = []
    vectors = []

    for phase, indices in phase_columns.items():
        cur_vec = H[:, indices].sum(axis=1)
        vectors.append(cur_vec)
        phases.append(phase)
        
    plot_stacked_curves(tp, vectors, phases)

    plt.legend()


def plot_stacked_curves(x, vectors, names):
    
    for i in range(len(vectors)):

        if i == 0:
            previous_stack = None
    
        cur_vector = vectors[i]
        name = names[i]

        previous_stack = plot_phase_stack(x, prev_vec=previous_stack, 
                                          cur_vec=cur_vector, name=name)
    