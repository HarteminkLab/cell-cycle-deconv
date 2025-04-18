import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def get_cell_vertices_with_spacing(center, spacing=(0.1, 0.1)):
	"""
	Get the vertices of a grid cell centered at the given point with custom spacing.
	
	Parameters:
	-----------
	center : list or tuple
		The center point [x, y] of the grid box
	spacing : tuple
		The (x, y) spacing of the grid
		
	Returns:
	--------
	vertices : list of tuples
		The four vertices of the grid cell in clockwise order
	"""
	x, y = center
	half_x = spacing[0] / 2
	half_y = spacing[1] / 2
	
	# Return vertices in clockwise order starting from bottom-left
	return [
		(x - half_x, y - half_y),  # bottom-left
		(x + half_x, y - half_y),  # bottom-right
		(x + half_x, y + half_y),  # top-right
		(x - half_x, y + half_y),  # top-left
	]

def build_edge_graph_with_spacing(centers, spacing=(0.1, 0.1)):
	"""
	Build a graph of vertices and edges from the given grid cell centers with custom spacing.
	
	Parameters:
	-----------
	centers : list of lists
		List of center points [[x1, y1], [x2, y2], ...]
	spacing : tuple
		The (x, y) spacing of the grid
		
	Returns:
	--------
	vertices : set
		Set of all vertices
	edge_counts : dict
		Dictionary mapping edges to their counts
	vertex_edges : dict
		Dictionary mapping vertices to their connected edges
	"""
	vertices = set()
	edge_counts = defaultdict(int)
	vertex_edges = defaultdict(list)
	
	# Process each grid cell
	for center in centers:
		cell_vertices = get_cell_vertices_with_spacing(center, spacing)
		
		# Add vertices to the set
		for vertex in cell_vertices:
			vertices.add(vertex)
		
		# Add edges to the graph
		for i in range(4):
			v1 = cell_vertices[i]
			v2 = cell_vertices[(i+1) % 4]
			
			# Ensure consistent edge representation (smaller vertex first)
			edge = (v1, v2) if v1 < v2 else (v2, v1)
			
			# Increment edge count
			edge_counts[edge] += 1
			
			# Add edge to vertex connections
			vertex_edges[v1].append(v2)
			vertex_edges[v2].append(v1)
	
	return vertices, edge_counts, vertex_edges

def find_boundary_edges(edge_counts):
	"""
	Find the boundary edges (those that appear exactly once).
	
	Parameters:
	-----------
	edge_counts : dict
		Dictionary mapping edges to their counts
		
	Returns:
	--------
	boundary_edges : list
		List of edges that form the boundary
	"""
	boundary_edges = []
	
	for edge, count in edge_counts.items():
		if count == 1:  # This is a boundary edge
			boundary_edges.append(edge)
	
	return boundary_edges

def trace_boundary_path(boundary_edges):
	"""
	Trace a path along the boundary edges to form a continuous loop.
	
	Parameters:
	-----------
	boundary_edges : list
		List of edges that form the boundary
		
	Returns:
	--------
	boundary_path : list
		Ordered list of vertices forming the boundary path
	"""
	# Convert edges to a more usable format for path tracing
	edge_graph = defaultdict(list)
	for v1, v2 in boundary_edges:
		edge_graph[v1].append(v2)
		edge_graph[v2].append(v1)
	
	# Start with any vertex on the boundary
	if not boundary_edges:
		return []
	
	start_vertex = boundary_edges[0][0]
	
	# Build the path
	path = [start_vertex]
	current = start_vertex
	visited_edges = set()
	
	while True:
		found_next = False
		
		for neighbor in edge_graph[current]:
			edge = (current, neighbor) if current < neighbor else (neighbor, current)
			
			if edge not in visited_edges:
				visited_edges.add(edge)
				path.append(neighbor)
				current = neighbor
				found_next = True
				break
		
		# If we've completed the loop or can't find next step
		if not found_next or (len(path) > 1 and path[-1] == start_vertex):
			break
	
	return path

def highlight_points_on_heatmap(ax, points, spacing=(0.1, 0.1), color='white', linewidth=2):
	"""
	Highlight selected points on a heatmap by drawing a merged boundary around them.
	
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
	boundary_path : list
		Ordered list of vertices forming the boundary path
	"""
	# Build the graph
	vertices, edge_counts, vertex_edges = build_edge_graph_with_spacing(points, spacing)
	
	# Find boundary edges
	boundary_edges = find_boundary_edges(edge_counts)
	
	# Trace the boundary path
	boundary_path = trace_boundary_path(boundary_edges)
	
	# Close the path if needed
	if boundary_path and boundary_path[0] != boundary_path[-1]:
		boundary_path.append(boundary_path[0])
	
	# Plot the boundary
	if boundary_path:
		x = [v[0] for v in boundary_path]
		y = [v[1] for v in boundary_path]
		ax.plot(x, y, color=color, linewidth=linewidth)
	
	return boundary_path
