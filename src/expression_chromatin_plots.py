import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def create_chromatin_expression_layout(
    n_rows=7,  # Number of rows in chromatin data (excluding annotation)
    figsize=(9, 5),  # Figure size
    chromatin_width_ratios=[1, 1, 1, 1],  # Width ratios for chromatin columns, total column width = 4
    expression_width=1.0,  # Width of expression plot relative to chromatin (1/4)
    cell_cycle_width=0.75,  # Width of cell cycle plot relative to chromatin (1/4)
    branch_spacing=1.0,  # Spacing between branches
    top_margin=0.95,  # Top margin for titles
    bottom_margin=0.05,  # Bottom margin
    height_ratios=None,  # Optional custom height ratios for rows
    annotation_height=0.75,  # Height of annotation row relative to data rows
    annotation_spacing=0.15  # Height of spacing between annotation and data rows
):
    """
    Creates a layout for chromatin and expression data visualization with gene annotations
    and cell cycle indicators.
    """
    # Calculate the number of columns needed
    n_branches = 3  # Recovery, Mother, Daughter
    cols_per_branch = 3  # Cell cycle, chromatin, and expression
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate width ratios for all columns including spacing
    width_ratios = []
    for branch in range(n_branches):
        # Add cell cycle column
        width_ratios.append(cell_cycle_width)
        # Add chromatin columns
        width_ratios.extend(chromatin_width_ratios)
        # Add expression column
        width_ratios.append(expression_width)
        # Add spacing (except after last branch)
        if branch < n_branches - 1:
            width_ratios.append(branch_spacing)
    
    # Create height ratios including annotation row and spacing row
    if height_ratios is None:
        height_ratios = [annotation_height, annotation_spacing] + [1] * n_rows
    else:
        height_ratios = [annotation_height, annotation_spacing] + height_ratios
    
    # Create GridSpec
    gs = gridspec.GridSpec(
        n_rows + 2,  # Add 2 for annotation row and spacing
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0,
        wspace=0
    )
    
    # Create axes for all plot types
    chromatin_axes = [[] for _ in range(n_branches)]
    expression_axes = []
    annotation_axes = []
    cell_cycle_axes = []
    
    # Calculate column indices for each branch
    col_idx = 0
    for branch in range(n_branches):
        # Cell cycle column index
        cycle_idx = col_idx
        # Chromatin start column index
        chrom_idx = col_idx + 1
        # Expression column index
        expr_idx = chrom_idx + len(chromatin_width_ratios)
        
        # Create annotation axis (spans only chromatin)
        ax = fig.add_subplot(gs[0, chrom_idx:expr_idx])
        annotation_axes.append(ax)
        
        # Create cell cycle axis (skipping annotation and spacing rows)
        ax = fig.add_subplot(gs[2:, cycle_idx])
        cell_cycle_axes.append(ax)
        
        # Create chromatin axes (starting after annotation and spacing rows)
        for row in range(n_rows):
            ax = fig.add_subplot(gs[row + 2, chrom_idx:expr_idx])
            chromatin_axes[branch].append(ax)
        
        # Create expression axis (skipping annotation and spacing rows)
        ax = fig.add_subplot(gs[2:, expr_idx])
        expression_axes.append(ax)
        
        # Move to next branch (including spacing)
        col_idx = expr_idx + 2 if branch < n_branches - 1 else expr_idx + 1
    
    # Adjust layout
    plt.subplots_adjust(
        top=top_margin,
        bottom=bottom_margin
    )
    
    return fig, chromatin_axes, expression_axes, annotation_axes, cell_cycle_axes
