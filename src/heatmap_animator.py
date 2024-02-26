
import matplotlib.pyplot as plt

from PIL import Image
from src.utils import mkdirs_safe
import pandas as pd


class PTRHeatmapAnimator:

	def __init__(self, data, chromatin_model):
		self.data = data
		self.chromatin_model = chromatin_model

	def plot_heatmap(self, index, save_path=None):

		data = self.data
		fig, (ax, ax0) = plt.subplots(1, 2, figsize=(12, 1.75))
		plt.subplots_adjust(top=0.85)
		cax = ax.imshow(self.data[index], cmap='magma_r', 
			aspect='auto', vmax=10, origin='lower', 
			extent=self.chromatin_model.bin_extents)
		ax.axvline(self.chromatin_model.computed_plus_one, c='black', lw=1)
		ax.set_xticks([])
		ax.set_yticks([])

		ax.set_title(f'Frame {index}')

		# ----------- cell cycle chart ---------------

		from src.model import color_for_key

		ax0 = plt.gca()

		phases = ['RG1', 'CG1', 'DG1', 'postG1']

		last_h_position_end = 0
		for i in range(len(phases)):
		    phase = phases[i]
		    hpositions = self.chromatin_model.config.get_Hpositions_for_phase(phase)
		    x_vals = [last_h_position_end, hpositions[-1]]
		    last_h_position_end = hpositions[-1]

		    ax0.plot(x_vals, [0, 0], color=color_for_key(phase),
		            lw=20, solid_capstyle='butt')
		    ax0.text((x_vals[0]+x_vals[1])/2, 0, phase, c='white', va='center', ha='center')

		ax0.set_xticks([])
		ax0.set_yticks([])
		ax0.axvline(index, c='gray', zorder=0)


		# --------------------------------------------

		if save_path is not None:
			plt.savefig(save_path)
			plt.close()


	def create_animation(self, save_path):
		frames_dir = 'tmp/frames'
		mkdirs_safe([frames_dir])

		# Generate and save each frame as an image
		frame_files = []

		frames = self.create_animation_order()

		animation_index = 0
		for _, row in frames.iterrows():
			frame = row.frame
			frame_file = f'{frames_dir}/frame_{animation_index}.png'
			self.plot_heatmap(frame, frame_file)
			frame_files.append(frame_file)
			animation_index += 1

		# Creating an animated GIF
		gif_path = save_path
		frames = [Image.open(frame) for frame in frame_files]
		frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, 
		duration=30, loop=0)


	def create_animation_order(self, phase_animation_order=['RG1', 'postG1',
		'CG1', 'postG1', 'DG1', 'postG1']):
		phases = []
		frames = []
		for phase in phase_animation_order:
			cur_frames = self.chromatin_model.config.get_Hpositions_for_phase(phase)
			phases = phases + [phase] * len(cur_frames)
			frames = frames + list(cur_frames)

		return pd.DataFrame({'frame': frames, 'phase': phases})
