

def layout_images_vertically(compositor, image_paths_arr, height_proportions=None, 
						   between_padding=20, margin=(20, 20), x_position=None,
						   offsets=None, image_keys=None, widths=None, heights=None, 
						   preserve_aspect_ratio=True):
	"""
	Layout images vertically with specified padding and margins, maximizing width.
	
	Parameters:
	-----------
	compositor : FigureCompositor
		The compositor instance to use for placing images
	image_paths_arr : list
		List of file paths to the images to place
	height_proportions : list, optional
		Relative height proportions for each image. If None, heights determined by aspect ratio.
	between_padding : int, optional
		Padding between adjacent images in logical pixels (default: 20)
	margin : tuple or int, optional
		Either (horizontal_margin, top_margin) or single value for both (default: (20, 20))
		Note: Bottom margin is flexible and not specified
	x_position : int, optional
		Fixed x-position for all images. If None, uses the horizontal margin.
	offsets : list, optional
		List of (x, y) tuples for position adjustments for each image. If None, no offsets.
	image_keys : list, optional
		Names/keys for the placed images. If None, uses filenames.
	widths : list, optional
		Explicit widths for each image. If None, uses maximum available width.
	heights : list, optional
		Explicit heights for each image. If None, heights are determined by width and aspect ratio.
	preserve_aspect_ratio : bool, optional
		Whether to maintain aspect ratio when resizing (default: True)
		
	Returns:
	--------
	dict
		Dictionary mapping image keys to their placement information
	"""
	if not image_paths_arr:
		return {}
	
	# Handle margin as tuple or single value
	if isinstance(margin, int):
		horizontal_margin = top_margin = margin
	else:
		horizontal_margin, top_margin = margin
	
	# Set default x_position if not specified
	if x_position is None:
		x_position = horizontal_margin
	
	# Calculate default width (maximum available width)
	default_width = compositor.logical_width - (2 * horizontal_margin)
	
	# Set default offsets if not specified
	if offsets is None:
		offsets = [(0, 0)] * len(image_paths_arr)
	elif len(offsets) != len(image_paths_arr):
		raise ValueError(f"Offsets length ({len(offsets)}) must match number of images ({len(image_paths_arr)})")
	
	# Set default image keys if not specified
	if image_keys is None:
		import os
		image_keys = [os.path.splitext(os.path.basename(path))[0] for path in image_paths_arr]
	elif len(image_keys) != len(image_paths_arr):
		raise ValueError(f"Image keys length ({len(image_keys)}) must match number of images ({len(image_paths_arr)})")
	
	# Set default widths if not specified
	if widths is None:
		widths = [default_width] * len(image_paths_arr)
	elif len(widths) != len(image_paths_arr):
		raise ValueError(f"Widths length ({len(widths)}) must match number of images ({len(image_paths_arr)})")
	
	# Set default heights if not specified
	if heights is None:
		heights = [None] * len(image_paths_arr)
	elif len(heights) != len(image_paths_arr):
		raise ValueError(f"Heights length ({len(heights)}) must match number of images ({len(image_paths_arr)})")
		
	# If height_proportions provided, adjust heights accordingly
	if height_proportions is not None:
		if len(height_proportions) != len(image_paths_arr):
			raise ValueError(f"Height proportions length ({len(height_proportions)}) must match number of images ({len(image_paths_arr)})")
		
		# If there are explicitly specified heights, use them as reference for proportions
		specified_heights = [h for h in heights if h is not None]
		
		if specified_heights:
			# Use the first specified height as reference
			reference_height = specified_heights[0] / height_proportions[heights.index(specified_heights[0])]
			
			# Adjust heights based on proportions
			for i, prop in enumerate(height_proportions):
				if heights[i] is None:  # Only adjust heights that aren't explicitly specified
					heights[i] = int(reference_height * prop)
		# If no heights specified, proportions are still useful for compositor.place_image
		# but won't be pre-calculated here
	
	# Place images
	placed_images = {}
	current_y = top_margin
	
	for i, (path, width, height, offset, key) in enumerate(zip(image_paths_arr, widths, heights, offsets, image_keys)):
		x_pos = x_position + offset[0]
		y_pos = current_y + offset[1]
		
		# Place the image
		img_info = compositor.place_image(
			path, 
			x_pos, 
			y_pos, 
			width=width, 
			height=height, 
			name=key,
			preserve_aspect_ratio=preserve_aspect_ratio
		)
		
		placed_images[key] = img_info
		
		# Update y position for next image based on the actual height of the placed image
		current_y += img_info['logical_size'][1] + between_padding
	
	return placed_images


def layout_images_horizontally(compositor, image_paths_arr, width_proportions=None, 
							 between_padding=20, margin=(20, 20), y_position=None,
							 offsets=None, image_keys=None, heights=None, preserve_aspect_ratio=True):
	"""
	Layout images horizontally with specified proportions, padding, and margins.
	
	Parameters:
	-----------
	compositor : FigureCompositor
		The compositor instance to use for placing images
	image_paths_arr : list
		List of file paths to the images to place
	width_proportions : list, optional
		Relative width proportions for each image. If None, all images get equal width.
	between_padding : int, optional
		Padding between adjacent images in logical pixels (default: 20)
	margin : tuple or int, optional
		Either (left_margin, top_margin) or single value for both (default: (20, 20))
	y_position : int, optional
		Fixed y-position for all images. If None, uses the top margin.
	offsets : list, optional
		List of (x, y) tuples for position adjustments for each image. If None, no offsets.
	image_keys : list, optional
		Names/keys for the placed images. If None, uses filenames.
	heights : list, optional
		Explicit heights for each image. If None, heights are determined by width and aspect ratio.
	preserve_aspect_ratio : bool, optional
		Whether to maintain aspect ratio when resizing (default: True)
		
	Returns:
	--------
	dict
		Dictionary mapping image keys to their placement information
	"""
	if not image_paths_arr:
		return {}
	
	# Handle margin as tuple or single value
	if isinstance(margin, int):
		left_margin = top_margin = margin
	else:
		left_margin, top_margin = margin
	
	# Set default y_position if not specified
	if y_position is None:
		y_position = top_margin
	
	# Set default width proportions if not specified
	if width_proportions is None:
		width_proportions = [1] * len(image_paths_arr)
	
	# Ensure width_proportions matches the number of images
	if len(width_proportions) != len(image_paths_arr):
		raise ValueError(f"Width proportions length ({len(width_proportions)}) must match number of images ({len(image_paths_arr)})")
	
	# Set default offsets if not specified
	if offsets is None:
		offsets = [(0, 0)] * len(image_paths_arr)
	elif len(offsets) != len(image_paths_arr):
		raise ValueError(f"Offsets length ({len(offsets)}) must match number of images ({len(image_paths_arr)})")
	
	# Set default image keys if not specified
	if image_keys is None:
		import os
		image_keys = [os.path.splitext(os.path.basename(path))[0] for path in image_paths_arr]
	elif len(image_keys) != len(image_paths_arr):
		raise ValueError(f"Image keys length ({len(image_keys)}) must match number of images ({len(image_paths_arr)})")
	
	# Calculate available width for images (canvas width minus margins and padding)
	available_width = compositor.logical_width - (2 * left_margin) - ((len(image_paths_arr) - 1) * between_padding)
	
	# Calculate total proportion units
	total_proportion = sum(width_proportions)
	
	# Calculate image widths based on proportions
	image_widths = [int((prop / total_proportion) * available_width) for prop in width_proportions]
	
	# Set default heights if not specified
	if heights is None:
		heights = [None] * len(image_paths_arr)
	elif len(heights) != len(image_paths_arr):
		raise ValueError(f"Heights length ({len(heights)}) must match number of images ({len(image_paths_arr)})")
	
	# Place images
	placed_images = {}
	current_x = left_margin
	
	for i, (path, width, height, offset, key) in enumerate(zip(image_paths_arr, image_widths, heights, offsets, image_keys)):
		x_pos = current_x + offset[0]
		y_pos = y_position + offset[1]
		
		# Place the image
		img_info = compositor.place_image(
			path, 
			x_pos, 
			y_pos, 
			width=width, 
			height=height, 
			name=key,
			preserve_aspect_ratio=preserve_aspect_ratio
		)
		
		placed_images[key] = img_info
		
		# Update x position for next image
		current_x += width + between_padding
	
	return placed_images


def place_image_below(compositor, image_path, img_key, 
				  vertical_padding=20, width=None, height=None,
				  offset=(0, 0), new_key=None, preserve_aspect_ratio=True):
	"""
	Place an image below an existing image with specified padding and optional adjustments.
	
	Parameters:
	-----------
	compositor : FigureCompositor
		The compositor instance to use
	image_path : str
		Path to the image to place
	img_key : str
		Key of the existing image to place the new image below
	vertical_padding : int, optional
		Vertical padding between the two images in logical pixels (default: 20)
	width : int or float, optional
		Width for the new image. If None, uses the same width as the reference image.
		If float < 1, interpreted as a proportion of the reference image width.
	height : int, optional
		Height for the new image. If None, height is determined by width and aspect ratio.
	offset : tuple, optional
		(x, y) offset adjustment from the calculated position (default: (0, 0))
	new_key : str, optional
		Key for the new image. If None, derives from the filename.
	preserve_aspect_ratio : bool, optional
		Whether to maintain aspect ratio when resizing (default: True)
		
	Returns:
	--------
	dict
		Information about the placed image
	"""
	# Check if the reference image exists
	if img_key not in compositor.placed_images:
		raise ValueError(f"Reference image with key '{img_key}' not found")
	
	# Get reference image info
	ref_img = compositor.placed_images[img_key]
	ref_x, ref_y = ref_img['logical_position']
	ref_width, ref_height = ref_img['logical_size']
	
	# Calculate x position (same as reference image by default)
	x_pos = ref_x + offset[0]
	
	# Calculate y position (below reference image plus padding)
	y_pos = ref_y + ref_height + vertical_padding + offset[1]
	
	# Determine width for the new image
	if width is None:
		# Use same width as reference image
		new_width = ref_width
	elif isinstance(width, float) and width < 1.0:
		# Interpret as proportion of reference width
		new_width = int(ref_width * width)
	else:
		# Use specified width directly
		new_width = width
	
	# Generate a key if not provided
	if new_key is None:
		import os
		new_key = os.path.splitext(os.path.basename(image_path))[0]
	
	# Place the image
	img_info = compositor.place_image(
		image_path,
		x_pos,
		y_pos,
		width=new_width,
		height=height,
		name=new_key,
		preserve_aspect_ratio=preserve_aspect_ratio
	)
	
	return img_info

def add_panel_labels_to_images(compositor, placed_images, labels=None, font_size=24, 
							 offset=(0, -40), font_type='bold', 
							 color=(0, 0, 0), background=None):
	"""
	Add panel labels (A, B, C, etc.) to a set of placed images.
	
	Parameters:
	-----------
	compositor : FigureCompositor
		The compositor instance to use
	placed_images : dict
		Dictionary of placed images from layout functions
	labels : list or str, optional
		Labels to use. If None, uses uppercase letters A, B, C...
		If a string, uses characters from the string.
	font_size : int, optional
		Font size for labels (default: 24)
	offset : tuple, optional
		(x, y) offset from top-left of each image (default: (0, -40))
	font_type : str, optional
		Font type to use (default: 'bold')
	color : tuple or str, optional
		Text color (default: black)
	background : tuple or str, optional
		Background color for labels. If None, no background is used.
		
	Returns:
	--------
	dict
		Dictionary mapping image keys to their label information
	"""
	if not placed_images:
		return {}
	
	image_keys = list(placed_images.keys())
	
	# Generate labels if not specified
	if labels is None:
		import string
		labels = list(string.ascii_uppercase[:len(image_keys)])
	elif isinstance(labels, str):
		labels = list(labels[:len(image_keys)])
	
	if len(labels) < len(image_keys):
		raise ValueError(f"Not enough labels ({len(labels)}) for images ({len(image_keys)})")
	
	# Add labels
	label_info = {}
	
	for i, (key, label) in enumerate(zip(image_keys, labels)):
		result = compositor.add_panel_label_to_image(
			key, 
			label, 
			offset=offset,
			font_size=font_size,
			color=color,
			font_type=font_type,
			background=background
		)
		
		if result:
			label_info[key] = {
				'label': label,
				'image_key': key
			}
	
	return label_info
