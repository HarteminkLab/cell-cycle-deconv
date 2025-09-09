#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure Compositor

This module provides functionality for stitching together multiple images
into composite figures with annotations such as panel labels.
"""

import os
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class FigureCompositor:
	"""
	A class for creating composite figures from multiple images.
	
	This class allows placing images on a canvas with precise positioning and
	adding annotations like panel labels (A, B, C, etc.). It supports scaling
	to render higher-resolution figures while maintaining the same positioning logic.
	"""
	
	def __init__(self, width: int, height: int, 
				 background_color: Union[str, Tuple[int, int, int]] = (255, 255, 255),
				 debug_mode: bool = False,
				 grid_size: int = 20,
				 grid_color: Union[str, Tuple[int, int, int]] = (220, 220, 220),
				 scale_factor: float = 4.0,
				 font_dir: str = "./fonts"):
		"""
		Initialize a new canvas for compositing images.
		
		Parameters:
		-----------
		width : int
			Width of the canvas in logical pixels
		height : int
			Height of the canvas in logical pixels
		background_color : str or tuple
			Background color as a PIL color name or RGB tuple (default: white)
		debug_mode : bool
			Whether to draw debug information (bounding boxes, coordinates) (default: False)
		grid_size : int
			Size of debug grid cells in logical pixels (default: 20)
		grid_color : str or tuple
			Color for the debug grid lines (default: light gray)
		scale_factor : float
			Factor to scale all dimensions and coordinates by (default: 1.0)
		font_dir : str
			Directory containing custom font files (default: "./fonts")
		"""
		self.scale_factor = scale_factor
		self.font_dir = font_dir
		
		# Store logical dimensions
		self.logical_width = width
		self.logical_height = height
		
		# Calculate actual dimensions after scaling
		self.width = int(width * scale_factor)
		self.height = int(height * scale_factor)
		
		self.background_color = background_color
		self.debug_mode = debug_mode
		self.logical_grid_size = grid_size
		self.grid_size = int(grid_size * scale_factor)
		self.grid_color = grid_color
		
		# Create blank canvas at the scaled size
		self.canvas = Image.new('RGB', (self.width, self.height), background_color)
		self.draw = ImageDraw.Draw(self.canvas)
		
		# Load fonts
		self._load_fonts()
		
		# Define debug colors
		self.debug_box_color = (255, 0, 0)  # Red for bounding boxes
		self.debug_text_color = (0, 0, 0)   # Black for text
		self.debug_bg_color = (255, 255, 200) # Light yellow for text background
		
		# Add debug grid if in debug mode
		if self.debug_mode:
			self._draw_debug_grid()
			
		# Track placed items for reference
		self.placed_images = {}
		self.annotations = {}
	
	def _load_fonts(self):
		"""
		Load custom fonts from the font directory.
		
		This method sets up font objects for regular, bold, semi-bold, and debug text
		at appropriate scaled sizes.
		"""
		# Define base font sizes
		base_font_size = 24
		small_font_size = 18
		debug_font_size = 12
		
		# Scale font sizes
		scaled_base_size = int(base_font_size * self.scale_factor)
		scaled_small_size = int(small_font_size * self.scale_factor)
		scaled_debug_size = int(debug_font_size * self.scale_factor)
		
		# Define font file paths
		font_paths = {
			'regular': os.path.join(self.font_dir, 'OpenSans-Regular.ttf'),
			'bold': os.path.join(self.font_dir, 'OpenSans-Bold.ttf'),
			'semi_bold': os.path.join(self.font_dir, 'OpenSans-SemiBold.ttf'),
			'medium': os.path.join(self.font_dir, 'OpenSans-Medium.ttf'),
			'light': os.path.join(self.font_dir, 'OpenSans-Light.ttf'),
			'extra_bold': os.path.join(self.font_dir, 'OpenSans-ExtraBold.ttf'),

			# Italic variants
			'italic': os.path.join(self.font_dir, 'OpenSans-Italic.ttf'),
			'bold_italic': os.path.join(self.font_dir, 'OpenSans-BoldItalic.ttf'),
			'semi_bold_italic': os.path.join(self.font_dir, 'OpenSans-SemiBoldItalic.ttf'),
		}
		
		# Initialize font dictionaries
		self.fonts = {}
		self.font_files = {}
		
		try:
			# Load all font types
			for font_type, font_path in font_paths.items():
				if os.path.exists(font_path):
					self.font_files[font_type] = font_path
				else:
					print(f"Warning: Font file not found: {font_path}")
			
			# Create regular-sized font instances
			self.fonts['regular'] = ImageFont.truetype(self.font_files.get('regular', self.font_files.get('medium', '')), scaled_base_size) if 'regular' in self.font_files else ImageFont.load_default()
			self.fonts['bold'] = ImageFont.truetype(self.font_files.get('bold', self.font_files.get('regular', '')), scaled_base_size) if 'bold' in self.font_files else ImageFont.load_default()
			self.fonts['semi_bold'] = ImageFont.truetype(self.font_files.get('semi_bold', self.font_files.get('regular', '')), scaled_base_size) if 'semi_bold' in self.font_files else ImageFont.load_default()
			
			# Create small-sized font instances
			self.fonts['small_regular'] = ImageFont.truetype(self.font_files.get('regular', self.font_files.get('medium', '')), scaled_small_size) if 'regular' in self.font_files else ImageFont.load_default()
			self.fonts['small_bold'] = ImageFont.truetype(self.font_files.get('bold', self.font_files.get('regular', '')), scaled_small_size) if 'bold' in self.font_files else ImageFont.load_default()
			self.fonts['small_semi_bold'] = ImageFont.truetype(self.font_files.get('semi_bold', self.font_files.get('regular', '')), scaled_small_size) if 'semi_bold' in self.font_files else ImageFont.load_default()
			
			# Create debug font instance
			self.fonts['debug'] = ImageFont.truetype(self.font_files.get('regular', ''), scaled_debug_size) if 'regular' in self.font_files else ImageFont.load_default()
			
			# Set primary fonts for backward compatibility
			self.font = self.fonts['regular']
			self.small_font = self.fonts['small_regular']
			self.debug_font = self.fonts['debug']
			
		except Exception as e:
			print(f"Error loading fonts: {e}")
			print("Falling back to default fonts")
			
			# Fall back to default fonts if there's an error
			self.font = ImageFont.load_default()
			self.small_font = ImageFont.load_default()
			self.debug_font = ImageFont.load_default()
			
			self.fonts = {
				'regular': self.font,
				'bold': self.font,
				'semi_bold': self.font,
				'small_regular': self.small_font,
				'small_bold': self.small_font,
				'small_semi_bold': self.small_font,
				'debug': self.debug_font
			}
	
	def get_font(self, font_type: str = 'regular', font_size: Optional[int] = None) -> ImageFont.FreeTypeFont:
		"""
		Get a font of the specified type and size.
		
		Parameters:
		-----------
		font_type : str
			Type of font to use ('regular', 'bold', 'semi_bold', etc.)
		font_size : int, optional
			Font size in logical pixels. If specified, creates a new font instance.
			Otherwise, returns a pre-loaded font.
			
		Returns:
		--------
		ImageFont.FreeTypeFont
			The requested font object
		"""
		if font_size is None:
			# Return pre-loaded font
			return self.fonts.get(font_type, self.fonts['regular'])
		
		# Scale the font size
		scaled_size = self._scale(font_size)
		
		try:
			# Get the corresponding font file
			if font_type in self.font_files:
				return ImageFont.truetype(self.font_files[font_type], scaled_size)
			else:
				# Fall back to regular if the requested type is not available
				if 'regular' in self.font_files:
					return ImageFont.truetype(self.font_files['regular'], scaled_size)
				# Fall back to default if no fonts are available
				return ImageFont.load_default()
		except Exception as e:
			print(f"Error loading font of type {font_type} and size {font_size}: {e}")
			return ImageFont.load_default()
	
	def _scale(self, value: Union[int, float]) -> int:
		"""
		Scale a value by the scale factor.
		
		Parameters:
		-----------
		value : int or float
			Value to scale
			
		Returns:
		--------
		int
			Scaled value rounded to nearest integer
		"""
		return int(value * self.scale_factor)
	
	def place_image(self, image_path: str, x: int, y: int, 
					width: Optional[int] = None, height: Optional[int] = None,
					name: Optional[str] = None, 
					preserve_aspect_ratio: bool = True,
					fill_transparent: bool = True,
					fill_color: Union[str, Tuple[int, int, int]] = (255, 255, 255),
					anchor: str = 'top_left') -> Dict:
		"""
		Place an image on the canvas at the specified position.
		
		Parameters:
		-----------
		image_path : str
			Path to the image file
		x : int
			X-coordinate for the top-left corner of the image (in logical coordinates)
		y : int
			Y-coordinate for the top-left corner of the image (in logical coordinates)
		width : int, optional
			Desired width to resize the image to (in logical coordinates)
		height : int, optional
			Desired height to resize the image to (in logical coordinates)
		name : str, optional
			Name identifier for the placed image
		preserve_aspect_ratio : bool, optional
			Whether to maintain aspect ratio when resizing (default: True)
		fill_transparent : bool, optional
			Whether to fill transparent areas with fill_color (default: True)
		fill_color : str or tuple, optional
			Color to use for filling transparent areas (default: white)
			
		Returns:
		--------
		dict
			Information about the placed image including original and scaled positions and dimensions
		"""
		try:
			# Open the image
			img = Image.open(image_path)
			
			# Handle transparency if the image has an alpha channel
			if fill_transparent and 'A' in img.getbands():
				# Create a new white background image
				bg = Image.new('RGB', img.size, fill_color)
				# Paste the image on the background using the alpha channel as a mask
				bg.paste(img, (0, 0), img.getchannel('A'))
				img = bg
			elif img.mode != 'RGB':
				# Convert to RGB mode if not already
				img = img.convert('RGB')
			
			# Scale logical coordinates to actual coordinates
			scaled_x = self._scale(x)
			scaled_y = self._scale(y)
			
			# Original image dimensions before any resizing
			original_width, original_height = img.size
			
			# Initialize variables for final dimensions
			final_width, final_height = original_width, original_height
			
			# Resize the image if requested
			if width is not None or height is not None:
				# Convert logical width/height to scaled width/height
				scaled_width = self._scale(width) if width is not None else None
				scaled_height = self._scale(height) if height is not None else None
				
				# Calculate new dimensions
				if preserve_aspect_ratio:
					if scaled_width is None:
						# Scale width proportionally to height
						scaled_width = int(original_width * (scaled_height / original_height))
					elif scaled_height is None:
						# Scale height proportionally to width
						scaled_height = int(original_height * (scaled_width / original_width))
					else:
						# Both width and height provided, determine which one to adjust
						orig_aspect = original_width / original_height
						target_aspect = scaled_width / scaled_height
						
						if orig_aspect > target_aspect:
							# Image is wider than target, constrain by width
							scaled_height = int(scaled_width / orig_aspect)
						else:
							# Image is taller than target, constrain by height
							scaled_width = int(scaled_height * orig_aspect)
				else:
					# If not preserving aspect ratio, ensure both width and height are set
					if scaled_width is None:
						scaled_width = original_width
					if scaled_height is None:
						scaled_height = original_height
				
				# Perform the resize
				img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
				final_width, final_height = img.size
				
				# Store logical dimensions for reference
				logical_final_width = width if width is not None else int(final_width / self.scale_factor)
				logical_final_height = height if height is not None else int(final_height / self.scale_factor)
			else:
				# If no resize requested, the final dimensions are the original ones
				logical_final_width = int(final_width / self.scale_factor)
				logical_final_height = int(final_height / self.scale_factor)

			# Adjust coordinates based on anchor point
			if anchor == 'top_right':
				x = x - logical_final_width
			elif anchor == 'bottom_left':
				y = y - logical_final_height
			elif anchor == 'bottom_right':
				x = x - logical_final_width
				y = y - logical_final_height
			elif anchor == 'center':
				x = x - logical_final_width // 2
				y = y - logical_final_height // 2

			# Update the scaled positions for storage
			# following adjustments to anchor
			scaled_x = self._scale(x)
			scaled_y = self._scale(y)
			
			# Paste the image onto the canvas at the scaled coordinates
			self.canvas.paste(img, (scaled_x, scaled_y))
			
			# Track the placed image
			if name is None:
				name = os.path.basename(image_path)

			# Store both logical and actual information
			img_info = {
				'path': image_path,
				'logical_position': (x, y),
				'position': (scaled_x, scaled_y),
				'logical_size': (logical_final_width, logical_final_height),
				'size': (final_width, final_height),
				'logical_bounds': (x, y, x + logical_final_width, y + logical_final_height),
				'bounds': (scaled_x, scaled_y, scaled_x + final_width, scaled_y + final_height)
			}
			
			self.placed_images[name] = img_info
			
			# Add debug bounding box if in debug mode
			if self.debug_mode:
				self._draw_debug_box(name)
			
			return img_info
			
		except Exception as e:
			print(f"Error placing image {image_path}: {e}")
			return {}
	
	def _draw_debug_box(self, image_name: str) -> None:
		"""
		Draw a debug bounding box and coordinates for an image.
		
		Parameters:
		-----------
		image_name : str
			Name identifier of the placed image
		"""
		if image_name not in self.placed_images:
			return
			
		info = self.placed_images[image_name]
		x1, y1, x2, y2 = info['bounds']
		logical_x1, logical_y1, logical_x2, logical_y2 = info['logical_bounds']
		
		# Draw rectangle
		self.draw.rectangle([x1, y1, x2, y2], outline=self.debug_box_color, width=max(1, int(2 * self.scale_factor)))
		
		# Add corner coordinates at top-left (showing logical coordinates)
		coord_text = f"({logical_x1},{logical_y1})"
		if hasattr(self.debug_font, 'getsize'):
			text_w, text_h = self.debug_font.getsize(coord_text)
		else:
			# For newer PIL versions
			left, top, right, bottom = self.debug_font.getbbox(coord_text)
			text_w, text_h = right - left, bottom - top
		
		# Draw background for text
		self.draw.rectangle([x1, y1, x1 + text_w + 4, y1 + text_h + 4], 
						  fill=self.debug_bg_color, outline=self.debug_box_color)
		
		# Draw text
		self.draw.text((x1 + 2, y1 + 2), coord_text, 
					  fill=self.debug_text_color, font=self.debug_font)
		
		# Add corner coordinates at bottom-right (showing logical coordinates)
		coord_text = f"({logical_x2},{logical_y2})"
		if hasattr(self.debug_font, 'getsize'):
			text_w, text_h = self.debug_font.getsize(coord_text)
		else:
			# For newer PIL versions
			left, top, right, bottom = self.debug_font.getbbox(coord_text)
			text_w, text_h = right - left, bottom - top
		
		# Draw background for text
		self.draw.rectangle([x2 - text_w - 4, y2 - text_h - 4, x2, y2], 
						  fill=self.debug_bg_color, outline=self.debug_box_color)
		
		# Draw text
		self.draw.text((x2 - text_w - 2, y2 - text_h - 2), coord_text, 
					  fill=self.debug_text_color, font=self.debug_font)
		
		# Add size information at top-right (showing logical dimensions)
		size_text = f"{logical_x2-logical_x1}×{logical_y2-logical_y1}"
		if hasattr(self.debug_font, 'getsize'):
			text_w, text_h = self.debug_font.getsize(size_text)
		else:
			# For newer PIL versions
			left, top, right, bottom = self.debug_font.getbbox(size_text)
			text_w, text_h = right - left, bottom - top
		
		# Draw background for text
		self.draw.rectangle([x2 - text_w - 4, y1, x2, y1 + text_h + 4], 
						  fill=self.debug_bg_color, outline=self.debug_box_color)
		
		# Draw text
		self.draw.text((x2 - text_w - 2, y1 + 2), size_text, 
					  fill=self.debug_text_color, font=self.debug_font)
	
	def add_panel_label(self, label: str, x: int, y: int, 
					   font_size: int = 24, color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
					   name: Optional[str] = None, font_type: str = 'bold',
					   text_anchor='lt') -> None:
		"""
		Add a panel label (e.g., 'A', 'B', 'C') to the canvas.
		
		Parameters:
		-----------
		label : str
			The label text to add
		x : int
			X-coordinate for the label position (in logical coordinates)
		y : int
			Y-coordinate for the label position (in logical coordinates)
		font_size : int, optional
			Font size for the label in logical pixels (default: 24)
		color : str or tuple, optional
			Text color as a PIL color name or RGB tuple (default: black)
		name : str, optional
			Name identifier for the annotation
		font_type : str, optional
			Type of font to use ('regular', 'bold', 'semi_bold') (default: 'bold')
		"""

		# Scale coordinates
		scaled_x = self._scale(x)
		scaled_y = self._scale(y)
		
		# Get appropriate font
		font = self.get_font(font_type, font_size)
		
		# Draw the label
		self.draw.text((scaled_x, scaled_y), label, fill=color, font=font,
			anchor=text_anchor)
		
		# Track the annotation with both logical and scaled values
		if name is None:
			name = f"label_{label}"
			
		self.annotations[name] = {
			'text': label,
			'logical_position': (x, y),
			'position': (scaled_x, scaled_y),
			'logical_font_size': font_size,
			'font_size': self._scale(font_size),
			'color': color,
			'font_type': font_type,
			'anchor': text_anchor
		}
	
	def save(self, output_path: str, quality: int = 95, dpi: Tuple[int, int] = (300, 300),
			 save_debug_version: bool = False) -> bool:
		"""
		Save the composite figure to a file.
		
		Parameters:
		-----------
		output_path : str
			Path to save the output image
		quality : int, optional
			JPEG quality (0-100) if saving as JPEG (default: 95)
		dpi : tuple of int, optional
			DPI information to embed in the image (default: (300, 300))
		save_debug_version : bool, optional
			Whether to save an additional debug version with bounding boxes (default: False)
			
		Returns:
		--------
		bool
			True if successfully saved, False otherwise
		"""
		# Ensure the directory exists
		os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
		
		# If we need a debug version and debug mode is off, create a new debug compositor
		if save_debug_version and not self.debug_mode:
			# Create a copy with debug enabled, same scale factor
			debug_comp = FigureCompositor(self.logical_width, self.logical_height, 
										 self.background_color, debug_mode=True,
										 grid_size=self.logical_grid_size, 
										 grid_color=self.grid_color,
										 scale_factor=self.scale_factor,
										 font_dir=self.font_dir)
			
			# Copy all placed images
			for name, info in self.placed_images.items():
				debug_comp.place_image(
					info['path'], 
					info['logical_position'][0], 
					info['logical_position'][1],
					info['logical_size'][0],
					info['logical_size'][1],
					name=name
				)
			
			# Copy all annotations
			for name, info in self.annotations.items():
				if 'text' in info:
					font_type = info.get('font_type', 'bold')
					debug_comp.add_panel_label(
						info['text'],
						info['logical_position'][0],
						info['logical_position'][1],
						info.get('logical_font_size', 24),
						info.get('color', (0, 0, 0)),
						name=name,
						font_type=font_type,
						text_anchor=info['anchor']
					)
			
			# Save the debug version
			debug_path = self._get_debug_path(output_path)
			debug_comp.canvas.save(debug_path, quality=quality, dpi=dpi)
			print(f"Debug figure saved to: {debug_path}")
		
		# If we're in debug mode and don't need a separate debug version,
		# save a clean version as well
		elif self.debug_mode and not save_debug_version:
			# Save the debug version (current canvas with debug annotations)
			debug_path = self._get_debug_path(output_path)
			self.canvas.save(debug_path, quality=quality, dpi=dpi)
			print(f"Debug figure saved to: {debug_path}")
			
			# Create a clean version with same scale factor
			clean_comp = FigureCompositor(self.logical_width, self.logical_height, 
										 self.background_color, debug_mode=False,
										 scale_factor=self.scale_factor,
										 font_dir=self.font_dir)
			
			# Copy all placed images
			for name, info in self.placed_images.items():
				clean_comp.place_image(
					info['path'], 
					info['logical_position'][0], 
					info['logical_position'][1],
					info['logical_size'][0],
					info['logical_size'][1],
					name=name
				)
			
			# Copy all annotations
			for name, info in self.annotations.items():
				if 'text' in info:
					font_type = info.get('font_type', 'bold')
					clean_comp.add_panel_label(
						info['text'],
						info['logical_position'][0],
						info['logical_position'][1],
						info.get('logical_font_size', 24),
						info.get('color', (0, 0, 0)),
						name=name,
						font_type=font_type,
						text_anchor=info['anchor']
					)
			
			# Save the clean version to the original path
			clean_comp.canvas.save(output_path, quality=quality, dpi=dpi)
			print(f"Clean figure saved to: {output_path}")
			
			return True
		
		# Normal case - just save the current canvas
		self.canvas.save(output_path, quality=quality, dpi=dpi)
		print(f"Figure saved to: {output_path}")
		
		return True
	
	def _get_debug_path(self, output_path: str) -> str:
		"""
		Create a debug version of the output path by inserting '_debug' before the extension.
		
		Parameters:
		-----------
		output_path : str
			Original output path
			
		Returns:
		--------
		str
			Path with '_debug' inserted before the extension
		"""
		base, ext = os.path.splitext(output_path)

		path_split = base.split('/')

		# Insert the debug folder into the debug path
		debug_path_split = (path_split[:-1]	+ ['Debug'] + path_split[-1:])
		debug_base_path = '/'.join(debug_path_split)

		return f"{debug_base_path}_debug{ext}"
		
	def _draw_debug_grid(self):
		"""
		Draw a debug grid on the canvas with light gray lines.
		Grid lines are drawn every self.grid_size pixels,
		with labels showing logical coordinates.
		"""
		# Draw vertical lines
		for i, x in enumerate(range(0, self.width + 1, self.grid_size)):
			self.draw.line([(x, 0), (x, self.height)], fill=self.grid_color, width=1)
			
			# Add coordinate labels for major lines (every 100 logical pixels)
			logical_x = i * self.logical_grid_size
			if logical_x % 100 == 0 and logical_x > 0:
				text = str(logical_x)
				
				if hasattr(self.debug_font, 'getsize'):
					text_w, text_h = self.debug_font.getsize(text)
				else:
					# For newer PIL versions
					left, top, right, bottom = self.debug_font.getbbox(text)
					text_w, text_h = right - left, bottom - top
				
				# Draw background for better visibility
				self.draw.rectangle([x - text_w // 2, 0, x + text_w // 2, text_h + 4], 
								  fill=self.debug_bg_color)
				
				# Draw text centered on the line
				self.draw.text((x - text_w // 2, 2), text, 
							 fill=self.debug_text_color, font=self.debug_font)
		
		# Draw horizontal lines
		for i, y in enumerate(range(0, self.height + 1, self.grid_size)):
			self.draw.line([(0, y), (self.width, y)], fill=self.grid_color, width=1)
			
			# Add coordinate labels for major lines (every 100 logical pixels)
			logical_y = i * self.logical_grid_size
			if logical_y % 100 == 0 and logical_y > 0:
				text = str(logical_y)
				
				if hasattr(self.debug_font, 'getsize'):
					text_w, text_h = self.debug_font.getsize(text)
				else:
					# For newer PIL versions
					left, top, right, bottom = self.debug_font.getbbox(text)
					text_w, text_h = right - left, bottom - top
				
				# Draw background for better visibility
				self.draw.rectangle([0, y - text_h // 2, text_w + 4, y + text_h // 2], 
								  fill=self.debug_bg_color)
				
				# Draw text centered on the line
				self.draw.text((2, y - text_h // 2), text, 
							 fill=self.debug_text_color, font=self.debug_font)
		
		# Draw origin marker
		origin_size = self._scale(5)
		self.draw.rectangle([0, 0, origin_size, origin_size], 
						  fill=(255, 0, 0))  # Red square at origin
						  
	def toggle_debug_mode(self, enabled: bool = None) -> bool:
		"""
		Toggle or set the debug mode.
		
		Parameters:
		-----------
		enabled : bool, optional
			If provided, sets debug mode to this value,
			otherwise toggles the current value
			
		Returns:
		--------
		bool
			The new debug mode state
		"""
		old_debug_mode = self.debug_mode
		
		if enabled is None:
			self.debug_mode = not self.debug_mode
		else:
			self.debug_mode = enabled
		
		# If debug mode was turned on (and wasn't already on), add the grid and debug boxes
		if self.debug_mode and not old_debug_mode:
			# Draw grid first so it's behind everything else
			self._draw_debug_grid()
			
			# Then add debug boxes to existing images
			for name in self.placed_images:
				self._draw_debug_box(name)
				
		return self.debug_mode
		
	def add_panel_label_to_image(self, image_name: str, label: str, 
								offset: Tuple[int, int] = (0, 0),
								font_size: int = 24, 
								color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
								font_type: str = 'bold',
								background: Optional[Union[str, Tuple[int, int, int]]] = None,
								bg_padding: int = 4,
								bg_opacity: int = 200,
								text_anchor='lt') -> bool:
		"""
		Add a panel label directly to a previously placed image.
		
		Parameters:
		-----------
		image_name : str
			Name identifier of the placed image to add the label to
		label : str
			The label text to add (e.g., 'A', 'B', 'C')
		offset : tuple of int, optional
			Offset from the top-left corner of the image in logical pixels (default: (0, 0))
		font_size : int, optional
			Font size for the label in logical pixels (default: 24)
		color : str or tuple, optional
			Text color as a PIL color name or RGB tuple (default: black)
		font_type : str, optional
			Type of font to use ('regular', 'bold', 'semi_bold') (default: 'bold')
		background : str or tuple, optional
			Background color for the label. If None, no background is drawn (default: None)
		bg_padding : int, optional
			Padding around the text for the background in logical pixels (default: 4)
		bg_opacity : int, optional
			Opacity for the background (0-255) if background is specified (default: 200)
			
		Returns:
		--------
		bool
			True if successfully added, False otherwise
		"""
		# Check if the image exists
		if image_name not in self.placed_images:
			print(f"Error: Image '{image_name}' not found.")
			return False
			
		# Get the image information
		img_info = self.placed_images[image_name]
		
		# Calculate label position in logical coordinates
		logical_x = img_info['logical_position'][0] + offset[0]
		logical_y = img_info['logical_position'][1] + offset[1]

		# Scale coordinates and font size
		scaled_x = self._scale(logical_x)
		scaled_y = self._scale(logical_y)
		scaled_bg_padding = self._scale(bg_padding) if background is not None else 0
		
		# Get appropriate font
		font = self.get_font(font_type, font_size)
		
		# Get text dimensions
		if hasattr(font, 'getsize'):
			text_w, text_h = font.getsize(label)
		else:
			# For newer PIL versions
			left, top, right, bottom = font.getbbox(label)
			text_w, text_h = right - left, bottom - top
		
		# Draw background if specified
		if background is not None:
			# Create a semi-transparent background
			if isinstance(background, tuple) and len(background) == 3:
				# Convert RGB to RGBA
				bg_color = (background[0], background[1], background[2], bg_opacity)
			else:
				# Use color as is (PIL will handle string color names)
				bg_color = background
				
			# Create a temporary transparent image for the background
			bg_img = Image.new('RGBA', (text_w + 2 * scaled_bg_padding, text_h + 2 * scaled_bg_padding), 
							  (0, 0, 0, 0))
			bg_draw = ImageDraw.Draw(bg_img)
			
			# Draw rounded rectangle background
			bg_draw.rounded_rectangle(
				[0, 0, text_w + 2 * scaled_bg_padding - 1, text_h + 2 * scaled_bg_padding - 1],
				radius=scaled_bg_padding, fill=bg_color
			)
			
			# Paste the background onto the canvas
			self.canvas.paste(bg_img, (scaled_x - scaled_bg_padding, scaled_y - scaled_bg_padding), bg_img)
		
		# Draw the label text
		self.draw.text((scaled_x, scaled_y), label, fill=color, font=font,
			anchor=text_anchor)
		
		# Track the annotation with both logical and scaled values
		label_name = f"label_{image_name}_{label}"
		self.annotations[label_name] = {
			'text': label,
			'logical_position': (logical_x, logical_y),
			'position': (scaled_x, scaled_y),
			'logical_font_size': font_size,
			'font_size': self._scale(font_size),
			'color': color,
			'font_type': font_type,
			'attached_to': image_name,
			'offset': offset,
			'anchor': text_anchor,
		}
		
		return True
