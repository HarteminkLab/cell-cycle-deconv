import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import warnings
from src.plot_helpers import plot_rect2


class AntisenseTranscriptVisualizer:
	"""
	Visualization class for antisense transcript detection results.
	
	Provides methods for validating and exploring transcript boundary calls
	through various plotting functions.
	"""
	
	def __init__(self, 
				 transcript_caller,
				 orf_plotter=None,
				 figsize_region: Tuple[int, int] = (18, 4)):
		"""
		Initialize the visualizer.
		"""
		
		self.caller = transcript_caller
		self.orf_plotter = orf_plotter
		self.figsize_region = figsize_region
		
		# Color schemes
		self.watson_color = 'blue'
		self.crick_color = 'red'
		self.watson_alpha = 0.1
		self.crick_alpha = 0.1
		
	def plot_region_validation(self, 
							 span: Tuple[int, int],
							 show_orf_annotations: bool = True,
							 show_transcript_boundaries: bool = True,
							 y_limit: Optional[float] = None,
							 title: Optional[str] = None):
		"""
		Plot RNA coverage and detected transcript boundaries for a genomic region.
		"""
		
		if self.caller.watson_pileups_df is None:
			raise ValueError("No pileup data available. Load data first.")
		
		# Validate span
		if span[0] < self.caller.chromosome_span[0] or span[1] > self.caller.chromosome_span[1]:
			warnings.warn(f"Requested span {span} extends beyond chromosome bounds {self.caller.chromosome_span}")
		
		# Create figure
		n_subplots = 2 if show_orf_annotations and self.orf_plotter else 1
		fig, axes = plt.subplots(n_subplots, 1, figsize=self.figsize_region, 
								gridspec_kw={'height_ratios': [1, 3] if n_subplots == 2 else [1]})
		
		if n_subplots == 1:
			axes = [axes]
		
		# Plot ORF annotations if requested
		subplot_idx = 0
		if show_orf_annotations and self.orf_plotter:
			self.orf_plotter.set_span_chrom(span, self.caller.chromosome)
			self.orf_plotter.plot_orf_annotations(axes[subplot_idx])
			axes[subplot_idx].set_xlim(span[0], span[1])
			subplot_idx += 1
		
		# Plot RNA coverage
		coverage_ax = axes[subplot_idx]
		self._plot_coverage_for_region(coverage_ax, span, y_limit)
		
		# Overlay transcript boundaries if requested
		if show_transcript_boundaries and self.caller.results_df is not None:
			self._overlay_transcript_boundaries(coverage_ax, span)
		
		# Set title
		if title is None:
			title = f"Chr {self.caller.chromosome}: {span[0]:,} - {span[1]:,} bp"
		fig.suptitle(title, fontsize=14, fontweight='bold')
		
		plt.tight_layout()
		return fig
	
	def plot_threshold_distribution(self, bins: int = 50):
		"""
		Plot distribution of pileup values with threshold lines.
		
		Parameters:
		-----------
		bins : int
			Number of histogram bins (default: 50)
		"""
		
		if self.caller.watson_pileups_df is None:
			raise ValueError("No pileup data available. Load data first.")
		
		# Compute average pileups and log transform
		watson_avg = self.caller.watson_pileups_df.mean(axis=0).values
		crick_avg = self.caller.crick_pileups_df.mean(axis=0).values
		
		watson_log = np.log2(watson_avg + 1)
		crick_log = np.log2(crick_avg + 1)
		combined_data = np.concatenate([watson_log, crick_log])
		
		eps_zero_cutoff = 0.1
		primary_threshold = self.caller.primary_threshold
		extension_threshold = self.caller.extension_threshold
		
		# Create plot
		fig, ax = plt.subplots(1, 1, figsize=(4, 3))
		
		# Plot histogram
		counts, bin_edges, patches = ax.hist(combined_data, bins=bins, alpha=0.7, 
										   color='steelblue', edgecolor='black', linewidth=0.5)
		
		# Add threshold lines
		ax.axvline(primary_threshold, color='red', linestyle='--', linewidth=2, 
				  label=f'Primary threshold: {primary_threshold:.3f}')
		ax.axvline(extension_threshold, color='orange', linestyle='--', linewidth=2,
				  label=f'Extension threshold: {extension_threshold:.3f}')
		
		# Labels and formatting
		ax.set_xlabel('Log2(Coverage + 1)', fontsize=12)
		ax.set_ylabel('Frequency', fontsize=12)
		ax.set_title(f'Distribution of Pileup Coverage\nChr {self.caller.chromosome}', 
					fontsize=14, fontweight='bold')
		ax.legend()
		
		# Add quantile information to title if using quantile-based thresholds
		quantile_info = f'Q{self.caller.primary_threshold_quantile:.0%} = {primary_threshold:.3f}, Q{self.caller.extension_threshold_quantile:.0%} = {extension_threshold:.3f}'
		ax.set_title(ax.get_title() + f'\n{quantile_info}', fontsize=12)
		
		plt.tight_layout()
		return fig
	
	# =====================================================================
	# INTERNAL PLOTTING METHODS
	# =====================================================================
	
	def _plot_coverage_for_region(self, ax, span, y_limit=None, title=None):
		"""Plot RNA coverage for a specific genomic region."""
		
		# Convert span to array indices
		start_idx = max(0, span[0] - self.caller.chromosome_span[0])
		end_idx = min(len(self.caller.watson_pileups_df.columns), 
					 span[1] - self.caller.chromosome_span[0])
		
		if start_idx >= end_idx:
			ax.text(0.5, 0.5, 'Region outside chromosome bounds', 
				   transform=ax.transAxes, ha='center', va='center')
			return
		
		# Get average coverage for the region
		watson_avg = self.caller.watson_pileups_df.iloc[:, start_idx:end_idx].mean(axis=0).values
		crick_avg = self.caller.crick_pileups_df.iloc[:, start_idx:end_idx].mean(axis=0).values
		
		# Convert to log scale
		watson_log = np.log2(watson_avg + 1)
		crick_log = np.log2(crick_avg + 1)
		
		# Create position array
		positions = np.arange(span[0], span[0] + len(watson_log))
		
		# Plot coverage
		ax.plot(positions, watson_log, color=self.watson_color, linewidth=1, 
			   label='Watson (+)')
		ax.plot(positions, -crick_log, color=self.crick_color, linewidth=1,
			   label='Crick (-)')
		
		# Formatting
		ax.set_xlim(span[0], span[1])
		ax.axhline(0, color='black', linewidth=1, alpha=0.5)
		
		if y_limit is not None:
			ax.set_ylim(-y_limit, y_limit)
		else:
			max_val = max(np.max(watson_log), np.max(crick_log))
			ax.set_ylim(-max_val * 1.1, max_val * 1.1)
		
		ax.set_xlabel('Genomic Position (bp)')
		ax.set_ylabel('Log2(Coverage + 1)')
		ax.grid(True, alpha=0.3)
		ax.legend()
		
		if title:
			ax.set_title(title)
	
	def _overlay_transcript_boundaries(self, ax, span):
		"""Overlay detected transcript boundaries on coverage plot."""

		results_df = self.caller.results_df
		
		# Filter transcripts that overlap with the span
		region_transcripts = results_df[
			(results_df.end > span[0]) & 
			(results_df.start < span[1])
		]
		
		# Get y-limits for rectangle plotting
		y_min, y_max = ax.get_ylim()
		
		# Plot transcript boundaries
		for _, transcript in region_transcripts.iterrows():
			start = max(transcript.start, span[0])
			end = min(transcript.end, span[1])

			overlapping_gene = transcript.overlapping_gene

			# If the transcript overlaps with a gene, gray out the color
			if not overlapping_gene == None:
				color = '#999'
			else:
				color = self.watson_color if transcript.strand == '+' else self.crick_color
			
			if transcript.strand == '+':
				# Watson transcript - plot in upper half
				rect = plt.Rectangle((start, 0), end - start, y_max,
								   alpha=self.watson_alpha, color=color,
								   linewidth=0)
				ax.add_patch(rect)
			else:
				# Crick transcript - plot in lower half  
				rect = plt.Rectangle((start, y_min), end - start, -y_min,
								   alpha=self.crick_alpha, color=color,
								   linewidth=0)
				ax.add_patch(rect)
	
