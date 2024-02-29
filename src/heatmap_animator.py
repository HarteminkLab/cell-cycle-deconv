
import matplotlib.pyplot as plt

from PIL import Image
from src.utils import mkdirs_safe
import pandas as pd


class FHeatmapAnimator:
	"""Class to the deconvolved F images as a heatmap animation"""

	def __init__(self, chromatin_model):
		self.chromatin_model = chromatin_model
		self.data = chromatin_model.get_f_images()

	def plot_heatmap(self, title, 
			index, animation_index, n, save_path=None, suptitle=None, boundaries=None,
			heatmap_ax=None, timeline_ax=None):

		data = self.data

		if heatmap_ax is None:

			fig, axs = plt.subplots(3, 2, figsize=(8, 3.5))
			plt.subplots_adjust(left=0.25, right=0.75, top=0.8)

			import numpy as np
			axs = np.array(axs).T

			heatmap_ax = axs[1][0]
			masked_heatmap_ax = axs[1][1]
			not_masked_heatmap_ax = axs[1][2]

			ptr_ax = axs[0][0]
			threshold_ax = axs[0][1]
			timeline_ax = axs[0][2]

			for ax in axs.flatten():
				ax.set_xticks([])
				ax.set_yticks([])

		def plot_im_hm(plt_ax, im, vmax=10, cmap='magma_r'):

			plt_ax.imshow(im, cmap=cmap, 
				aspect='auto', vmax=vmax, origin='lower', 
				extent=self.chromatin_model.bin_extents)
			plt_ax.axvline(self.chromatin_model.computed_plus_one, 
				c='black', lw=1, alpha=0.25)
			plt_ax.set_xticks([])
			plt_ax.set_yticks([])

		cur_img = self.data[index]
		plot_im_hm(heatmap_ax, cur_img)

		def set_ylabel_ax(ax, label, labelposition='left'):
			ha = 'right' if labelposition == 'left' else 'left'
			ax.set_ylabel(label, rotation=0, ha=ha, labelpad=10)
			ax.yaxis.set_label_position(labelposition)


		set_ylabel_ax(heatmap_ax, "F", 'right')
		heatmap_ax.set_title(f"{title} {animation_index} / {n}", ha='center')

		# --------- ptr -----------

		ptr_img = self.chromatin_model.f_ptrs.reshape(self.chromatin_model.image_shape)
		plot_im_hm(ptr_ax, ptr_img, vmax=10, cmap='Blues')
		set_ylabel_ax(ptr_ax, "PTR")

		# ---------- threshold ----------

		from src.ptr_analysis_plotter import threshold_img

		threshold_ptr_img = threshold_img(ptr_img)
		plot_im_hm(threshold_ax, threshold_ptr_img, vmax=1, cmap='Blues')
		set_ylabel_ax(threshold_ax, "PTR threshold\nmask")

		# --------- masked/not masked animation ------------

		masked_img = cur_img * threshold_ptr_img
		plot_im_hm(masked_heatmap_ax, masked_img)
		set_ylabel_ax(masked_heatmap_ax, "F & mask", 'right')

		not_masked_img = cur_img*(1-threshold_ptr_img)
		plot_im_hm(not_masked_heatmap_ax, not_masked_img)
		set_ylabel_ax(not_masked_heatmap_ax, "F & ~mask", 'right')

		# ----------- cell cycle chart ---------------

		from src.model import color_for_key
		
		last_h_position_end = 0
		for _, boundary in boundaries.iterrows():
			phase = boundary.phase
			x_vals = [boundary.start, boundary.end]
			timeline_ax.plot(x_vals, [0, 0], color=color_for_key(phase),
					lw=20, solid_capstyle='butt')
			timeline_ax.text((x_vals[0]+x_vals[1])/2, 0, phase, c='white', va='center', ha='center')
		timeline_ax.set_xticks([])
		timeline_ax.set_yticks([])
		timeline_ax.axvline(animation_index, c='gray', zorder=0, alpha=0.5)

		set_ylabel_ax(timeline_ax, "Animation\nkey")

		# --------------------------------------------

		plt.suptitle(suptitle, fontsize=19)

		if save_path is not None:
			plt.savefig(save_path, dpi=150)
			plt.close()


	def create_animation(self, save_path):
		frames_dir = 'tmp/frames'
		mkdirs_safe([frames_dir])

		# Generate and save each frame as an image
		frame_files = []

		frames = self.create_animation_order()

		animation_index = 0

		boundaries = self.get_frame_boundaries(frames)
		n = len(frames)

		for _, row in frames.iterrows():
			frame = row.frame
			frame_file = f'{frames_dir}/frame_{animation_index}.png'
			title = f"{row.phase}"
			self.plot_heatmap(title, frame, animation_index, n,
				frame_file, boundaries=boundaries, suptitle=self.chromatin_model.gene_title())
			frame_files.append(frame_file)
			animation_index += 1

		# Creating an animated GIF
		gif_path = save_path
		frames = [Image.open(frame) for frame in frame_files]
		frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, 
		duration=45, loop=0)


	def create_animation_order(self, phase_animation_order=['CG1', 'postG1', 'DG1', 'postG1']):
		phases = []
		frames = []
		for phase in phase_animation_order:
			cur_frames = self.chromatin_model.config.get_Hpositions_for_phase(phase)
			phases = phases + [phase] * len(cur_frames)
			frames = frames + list(cur_frames)

		return pd.DataFrame({'frame': frames, 'phase': phases})

	def get_frame_boundaries(self, frames):
		"""Get frame boundaries for plotting the animation timeline"""
		start = 0
		end = 0
		phase = frames.iloc[0].phase
		starts = [start]
		ends = []
		phases = []

		for index, row in frames.iterrows():
			if phase != row.phase:
				end = index-1
				start = index
				ends.append(end)
				starts.append(start)
				phases.append(phase)
				phase = row.phase
		phases.append(phase)
		ends.append(index)

		animation_frame_boundaries = pd.DataFrame({'start': starts, 'end': ends, 'phase': phases})
		return animation_frame_boundaries
