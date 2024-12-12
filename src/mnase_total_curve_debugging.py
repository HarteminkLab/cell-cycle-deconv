
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_G_img_data(G_img_data):
	fig, axs = plt.subplots(8, 2, figsize=(3, 6))
	axs = np.array(axs).T.flatten()
	for i in range(G_img_data.shape[0]):
		ax = axs[i]

		normalized = G_img_data[i] / G_img_data[i].sum() * 1000

		ax.imshow(normalized,
			  origin='lower', aspect='auto', vmin=0, vmax=10, cmap='magma_r')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_ylabel(f"{i}'")
	plt.subplots_adjust(hspace=0.5)

def plot_deconvolved(genome_deconvolution):
	
	t_indices = genome_deconvolution.config1.get_Hpositions_for_branch('t')
	i_indices = genome_deconvolution.config1.get_Hpositions_for_branch('i')

	f = genome_deconvolution.combined_model.deconvolved_f_value.copy()

	G1_img_data = genome_deconvolution.combined_model.chrom1_model\
		.deconv_hist_unflattened+1
	G2_img_data = genome_deconvolution.combined_model.chrom2_model\
		.deconv_hist_unflattened+1

	plt.figure(figsize=(9, 4))

	plt.subplot(1, 4, 1)
	plt.imshow(G1_img_data.reshape((G1_img_data.shape[0], -1)).T, aspect='auto')
	plt.ylim(60, 110)

	plt.subplot(1, 4, 2)
	plt.imshow(G1_img_data.reshape((G1_img_data.shape[0], -1)).T, aspect='auto')
	plt.ylim(60, 110)

	plt.subplot(1, 4, 3)
	plt.imshow(f[i_indices].T, interpolation='none', aspect='auto')
	plt.ylim(60, 110)

	plt.subplot(1, 4, 4)
	plt.imshow(f[t_indices].T, interpolation='none', aspect='auto')
	plt.ylim(60, 110)


def plot_f_genome_deconvolution_nuc_reads(genome_deconvolution):
	
	shp = genome_deconvolution.combined_model.deconvolved_f_value.shape
	t_indices = genome_deconvolution.config1.get_Hpositions_for_branch('t')
	i_indices = genome_deconvolution.config1.get_Hpositions_for_branch('i')
	
	f_imgs = genome_deconvolution.combined_model.\
		deconvolved_f_value.reshape((shp[0], 26, -1))
	nuc_reads = f_imgs[:, 12:18, :].sum(axis=1).T

	plt.figure(figsize=(9, 4))
	plt.subplot(2, 3, 1)
	plt.imshow(nuc_reads[:, i_indices], aspect='auto')
	plt.title("Initial")
	
	plt.subplot(2, 3, 2)
	plt.imshow(nuc_reads[:, t_indices], aspect='auto')
	plt.yticks([])
	plt.title("Top")
	
	plt.subplot(2, 3, 3)
	plt.plot(nuc_reads[:, i_indices].sum(axis=0), label="Initial")
	plt.plot(nuc_reads[:, t_indices].sum(axis=0), label="Top")
	plt.legend()
	plt.ylim(0, 600)


	f = genome_deconvolution.combined_model.\
		deconvolved_f_value
	normalized_f = f / f.mean(axis=1).reshape((-1, 1))
	normalized_f = normalized_f.reshape((shp[0], 26, -1))

	nuc_reads = normalized_f[:, 12:18, :].sum(axis=1).T

	# Normalized reads 
	plt.subplot(2, 3, 4)
	plt.imshow(nuc_reads[:, i_indices], aspect='auto')
	plt.title("Initial")
	
	plt.subplot(2, 3, 5)
	plt.imshow(nuc_reads[:, t_indices], aspect='auto')
	plt.yticks([])
	plt.title("Top")
	
	plt.subplot(2, 3, 6)
	plt.plot(nuc_reads[:, i_indices].sum(axis=0), label="Initial")
	plt.plot(nuc_reads[:, t_indices].sum(axis=0), label="Top")
	plt.legend()
	plt.ylim(0, 500)




def plot_deconvolved_f_curve(genome_deconvolution, index):

	f = genome_deconvolution.combined_model.deconvolved_f_value.copy()
	
	t_indices = genome_deconvolution.config1.get_Hpositions_for_branch('t')
	i_indices = genome_deconvolution.config1.get_Hpositions_for_branch('i')
	
	plt.figure(figsize=(9, 2))
	plt.subplot(1, 3, 1)
	model = genome_deconvolution.combined_model
	predicted_G1 = model.H1 @ f
	predicted_G2 = model.H2 @ f
	
	plt.plot(model.G1[:, index], label='G1', c='red')
	plt.plot(predicted_G1[:, index], label='predicted G1', c='red', ls='dotted')
	
	plt.plot(model.G2[:, index], label='G2', c='blue')
	plt.plot(predicted_G2[:, index], label='predicted G2', c='blue', ls='dotted')
		
	plt.title("Example Raw")
	plt.legend()
	
	plt.subplot(1, 3, 2)
	plt.plot(f[t_indices, index], label='Top')
	plt.plot(f[i_indices, index], label="Initial")
	plt.title("Example deconvolution")
	plt.legend()
	
	plt.subplot(1, 3, 3)
	
	plt.plot(f[t_indices].mean(axis=1), label="Top")
	plt.plot(f[i_indices].mean(axis=1), label="Initial")
	
	plt.title("Sum")
	plt.legend()

	# Removing mirroring helps in creating a smooth curve... 
	# Now there is the effect of doubling
	# Maybe this is better, but there is still an artifact around the CG1 to S seam
	# 
	# Probably still related to the same recovery-S-G2M scheme
	# It's worth trying a deconvolution with the new model


def show_gif(gif_path):
	from IPython.display import Image
	import time

	gif_url = f"{gif_path}?{time.time()}" # bypass cached image
	display(Image(data=open(gif_path,'rb').read(), format='png'))
		
def create_animation(indices, gif_save_path, frames_dir,
	prefix):
	from PIL import Image

	frames_dir = 'output/tmp/'
	frame_files = []

	for animation_index in indices:
		frame_file = f'{frames_dir}/{prefix}{animation_index}.png'
		frame_files.append(frame_file)

	frame_duration_s = 0.5
	ms = 1000
	loops = 0

	durations = list(np.repeat(frame_duration_s*ms, len(frame_files)))

	frames = [Image.open(frame) for frame in frame_files]
	frames[0].save(gif_save_path, format='GIF', append_images=frames[1:], save_all=True, 
		duration=durations, loop=loops)


def create_weighted_smoothing_curve(config, plot=False):

	from scipy.stats import norm

	f_i = config.get_Hpositions_for_branch('i')
	f_t = config.get_Hpositions_for_branch('t')
	# Idea: It appears the bridge between
	# CG1 and postG1 isn't smoothed enough compared to 
	# RG1 and postG1. Is there a way we can add extra weighting on the
	# importance of smoothing this joint?

	rg1_indices = config.get_Hpositions_for_phase('RG1')
	cg1_indices = config.get_Hpositions_for_phase('CG1')
	postG1_indices = config.get_Hpositions_for_phase('postG1')

	f_it = np.concatenate([f_i, f_t])

	absolute_indices = np.arange(len(f_it))

	rg1_s_boundary = len(rg1_indices)
	cg1_s_boundary = len(rg1_indices)+len(postG1_indices)+len(cg1_indices)
	postG1_cg1_boundary = len(rg1_indices)+len(postG1_indices)


	scale = 10
	factor = 2
	rg1_s_weight = norm.pdf(absolute_indices, loc=rg1_s_boundary, scale=scale)*factor
	cg1_s_weight = norm.pdf(absolute_indices, loc=cg1_s_boundary, scale=scale)*factor
	postg1_cg1_weight = norm.pdf(absolute_indices, loc=postG1_cg1_boundary, scale=scale)*factor

	weight_scaling_vector = rg1_s_weight+cg1_s_weight+postg1_cg1_weight+1

	if plot:
		plt.plot(absolut_indices, f_it)

		ax = plt.gca()
		ax2 = ax.twinx()
		ax2.plot(weight_scaling_vector)
		ax2.set_ylim(0, 5)

	return weight_scaling_vector

