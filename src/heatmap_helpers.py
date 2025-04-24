import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_grid_from_points(points, spacing):
    """
    Generate a binary grid representation from the given points.
    
    Parameters:
    -----------
    points : list of lists
        List of center points [[x1, y1], [x2, y2], ...]
    spacing : tuple
        The (x, y) spacing of the grid
        
    Returns:
    --------
    grid : 2D numpy array
        Binary grid where 1 indicates a highlighted cell
    x_coords : numpy array
        X-coordinates for grid cells
    y_coords : numpy array
        Y-coordinates for grid cells
    """
    if not points:
        return np.array([]), np.array([]), np.array([])
    
    # Extract x and y coordinates
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    
    # Determine grid boundaries
    min_x, max_x = min(x) - spacing[0], max(x) + spacing[0]
    min_y, max_y = min(y) - spacing[1], max(y) + spacing[1]
    
    # Create coordinate arrays with spacing
    x_coords = np.arange(min_x, max_x + 0.1*spacing[0], spacing[0])
    y_coords = np.arange(min_y, max_y + 0.1*spacing[1], spacing[1])
    
    # Create empty grid
    grid = np.zeros((len(y_coords), len(x_coords)))
    
    # Mark cells in grid
    for point in points:
        # Find closest grid cell
        i = int(round((point[1] - min_y) / spacing[1]))
        j = int(round((point[0] - min_x) / spacing[0]))
        
        # Ensure within bounds
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            grid[i, j] = 1
    
    return grid, x_coords, y_coords

def trace_contour(grid, x_coords, y_coords):
    """
    Trace the contour of the highlighted region in the grid.
    
    Parameters:
    -----------
    grid : 2D numpy array
        Binary grid where 1 indicates a highlighted cell
    x_coords : numpy array
        X-coordinates for grid cells
    y_coords : numpy array
        Y-coordinates for grid cells
        
    Returns:
    --------
    contour : list of tuples
        List of (x, y) coordinates that form the contour
    """
    if grid.size == 0:
        return []
    
    # Pad the grid with zeros to handle boundary cases
    padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
    
    # Initialize contour
    contour = []
    
    # Define cell corner offsets (relative to bottom-left corner)
    corner_offsets = [
        (0, 0),  # bottom-left
        (1, 0),  # bottom-right
        (1, 1),  # top-right
        (0, 1)   # top-left
    ]
    
    # Helper function to get cell corners in world coordinates
    def get_cell_corners(i, j):
        x = x_coords[j]
        y = y_coords[i]
        half_x = (x_coords[1] - x_coords[0]) / 2
        half_y = (y_coords[1] - y_coords[0]) / 2
        
        return [
            (x - half_x, y - half_y),  # bottom-left
            (x + half_x, y - half_y),  # bottom-right
            (x + half_x, y + half_y),  # top-right
            (x - half_x, y + half_y)   # top-left
        ]
    
    # Find boundary segments
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:  # This is a highlighted cell
                cell_corners = get_cell_corners(i, j)
                
                # Check each edge of the cell
                for k in range(4):
                    ni, nj = i, j
                    
                    # Get neighboring cell based on edge
                    if k == 0:  # Bottom edge
                        ni = i - 1
                    elif k == 1:  # Right edge
                        nj = j + 1
                    elif k == 2:  # Top edge
                        ni = i + 1
                    elif k == 3:  # Left edge
                        nj = j - 1
                    
                    # Check if this edge is on the boundary
                    if ni < 0 or ni >= grid.shape[0] or nj < 0 or nj >= grid.shape[1] or grid[ni, nj] == 0:
                        # This edge is on the boundary, add its vertices to the contour
                        v1_idx = k
                        v2_idx = (k + 1) % 4
                        
                        contour.append((cell_corners[v1_idx], cell_corners[v2_idx]))
    
    # Now connect the boundary segments into a continuous path
    if not contour:
        return []
    
    # Sort segments to form a continuous path
    connected_path = []
    remaining_segments = contour.copy()
    
    # Start with the first segment
    current_segment = remaining_segments.pop(0)
    connected_path.extend([current_segment[0], current_segment[1]])
    current_point = current_segment[1]
    
    # Set a maximum number of iterations to prevent infinite loops
    max_iterations = len(contour) * 2
    iterations = 0
    
    # Continue connecting segments
    while remaining_segments and iterations < max_iterations:
        iterations += 1
        found = False
        
        # Find a segment that connects to the current point
        for i, segment in enumerate(remaining_segments):
            # Check both endpoints of the segment
            if np.allclose(segment[0], current_point, atol=1e-10):
                connected_path.append(segment[1])
                current_point = segment[1]
                remaining_segments.pop(i)
                found = True
                break
            elif np.allclose(segment[1], current_point, atol=1e-10):
                connected_path.append(segment[0])
                current_point = segment[0]
                remaining_segments.pop(i)
                found = True
                break
        
        if not found and remaining_segments:
            # If no connecting segment found, start a new path with the next segment
            current_segment = remaining_segments.pop(0)
            connected_path.extend([current_segment[0], current_segment[1]])
            current_point = current_segment[1]
    
    return connected_path

def simplify_contour(contour, tolerance=1e-10):
    """
    Simplify the contour by removing duplicate and nearly duplicate points.
    
    Parameters:
    -----------
    contour : list of tuples
        List of (x, y) coordinates that form the contour
    tolerance : float
        Tolerance for considering points duplicates
        
    Returns:
    --------
    simplified : list of tuples
        Simplified contour
    """
    if not contour:
        return []
    
    simplified = [contour[0]]
    
    for point in contour[1:]:
        last = simplified[-1]
        # Check if this point is significantly different from the last one
        if not (abs(point[0] - last[0]) < tolerance and abs(point[1] - last[1]) < tolerance):
            simplified.append(point)
    
    # Ensure the contour is closed
    first, last = simplified[0], simplified[-1]
    if not (abs(first[0] - last[0]) < tolerance and abs(first[1] - last[1]) < tolerance):
        simplified.append(simplified[0])
    
    return simplified

def highlight_points_on_heatmap(ax, points, spacing=(0.1, 0.1), color='white', linewidth=2):
    """
    Highlight selected points on a heatmap by drawing a boundary around them.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis of the existing heatmap
    points : list of lists
        List of points [[x1, y1], [x2, y2], ...] to highlight
    spacing : tuple
        The (x, y) spacing of the heatmap grid
    color : str
        Color for the highlight boundary
    linewidth : float
        Width of the boundary line
        
    Returns:
    --------
    contour : list of tuples
        List of (x, y) coordinates that form the contour
    """
    # Generate grid representation
    grid, x_coords, y_coords = generate_grid_from_points(points, spacing)
    
    # Trace contour
    raw_contour = trace_contour(grid, x_coords, y_coords)
    
    # Simplify contour
    contour = simplify_contour(raw_contour)
    
    # Plot the contour
    if contour:
        x = [p[0] for p in contour]
        y = [p[1] for p in contour]
        ax.plot(x, y, color=color, linewidth=linewidth)
    
    return contour