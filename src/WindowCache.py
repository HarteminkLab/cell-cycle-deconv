
from collections import OrderedDict


# Simple LRU cache for window data
class WindowCache:
	def __init__(self, max_size=3):
		"""
		Initialize a cache for genomic window data with a maximum size.
		
		Args:
			max_size: Maximum number of windows to keep in the cache (default: 3)
		"""
		self.max_size = max_size
		self.cache = OrderedDict()  # OrderedDict maintains insertion order
	
	def get(self, key):
		"""
		Get a window from the cache if it exists.
		
		Args:
			key: Tuple of (chromosome, window_start, window_end)
			
		Returns:
			The window data if it exists in the cache, None otherwise
		"""
		if key in self.cache:
			# Move to the end to mark as most recently used
			value = self.cache.pop(key)
			self.cache[key] = value
			return value
		return None
	
	def put(self, key, value):
		"""
		Add a window to the cache.
		
		Args:
			key: Tuple of (chromosome, window_start, window_end)
			value: The window data to cache
		"""
		# If key already exists, remove it first
		if key in self.cache:
			self.cache.pop(key)
		
		# Add new item
		self.cache[key] = value
		
		# Remove oldest item if cache exceeds max size
		if len(self.cache) > self.max_size:
			self.cache.popitem(last=False)
	
	def clear(self):
		"""Clear the entire cache."""
		self.cache.clear()

# Simple LRU cache for window data
class WindowCache:
	def __init__(self, max_size=3):
		"""
		Initialize a cache for genomic window data with a maximum size.
		
		Args:
			max_size: Maximum number of windows to keep in the cache (default: 3)
		"""
		self.max_size = max_size
		self.cache = OrderedDict()  # OrderedDict maintains insertion order
	
	def get(self, key):
		"""
		Get a window from the cache if it exists.
		
		Args:
			key: Tuple of (chromosome, window_start, window_end)
			
		Returns:
			The window data if it exists in the cache, None otherwise
		"""
		if key in self.cache:
			# Move to the end to mark as most recently used
			value = self.cache.pop(key)
			self.cache[key] = value
			return value
		return None
	
	def put(self, key, value):
		"""
		Add a window to the cache.
		
		Args:
			key: Tuple of (chromosome, window_start, window_end)
			value: The window data to cache
		"""
		# If key already exists, remove it first
		if key in self.cache:
			self.cache.pop(key)
		
		# Add new item
		self.cache[key] = value
		
		# Remove oldest item if cache exceeds max size
		if len(self.cache) > self.max_size:
			self.cache.popitem(last=False)
	
	def clear(self):
		"""Clear the entire cache."""
		self.cache.clear()