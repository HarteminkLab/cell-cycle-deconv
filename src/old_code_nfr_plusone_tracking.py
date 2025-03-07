def find_origin_p1_and_m1_nucleosome_position(self, find_p1=True):
		"""
		Identify the +1 and -1 nucleosome and track its position
		"""

		from src.chrom_img_segment_selector import translate_span_for_bins
		from src.chromatin_metric_tracking import ChromatinMetricTracking
		from src.chromatin_metrics import fragment_lengths_definitions
		from src.global_config import GlobalConstants

		# Center on the middle of the window (centered on the origin site)
		center_pos = self.origin.pos

		img_data = self.get_f_images()

		# Fragment lengths
		sm_lens, med_lens, nuc_lens = fragment_lengths_definitions()
		print("Nucleosomal fragment length range: ", nuc_lens)

		# Select a largish window of fragments for origins, such that we can retrieve the entirety of what
		# appears to be origin fragments
		origin_frag_lens = sm_lens[0]+GlobalConstants.BIN_HEIGHT, sm_lens[1]+GlobalConstants.BIN_HEIGHT
		print("Origin fragment length range: ", origin_frag_lens)

		# todo: Add an additional step here, in which we expand the search range for each
		# metric, and automate narrowing the search tighter based peak occupancy and a window around the peak
		nucleosome_movement_span = 143
		origin_occ_span = 99

		# Wide search span that will automatically be narrowed down
		p1_search_span = (0, 352)
		m1_search_span = (-352, 0)
		origin_span = (-121, 121)

		# Override to refine search window for selected origins
		if self.origin.ars_name == 'ARS423':
			p1_search_span = (0, 242)
			m1_search_span = (-242, 0)
		elif self.origin.ars_name == 'ARS1623':
			p1_search_span = (0, 242)
			m1_search_span = (-142, 0)
			print(m1_search_span)

		# Currently we allow the search span to be any genomic position, but the bins restrict us
		# to the bin width (24 bp), so we need to round to the nearest 24 bp bin
		is_crick = (self.origin.strand == '-')
		updated_p1_span = translate_span_for_bins(center_pos, p1_search_span, 
			flip=is_crick)
		updated_m1_span = translate_span_for_bins(center_pos, m1_search_span, 
			flip=is_crick)
		updated_origin_span = translate_span_for_bins(center_pos, origin_span, 
			flip=is_crick)

		print(f"The +1 span for tracking is:", updated_p1_span, " length: ", updated_p1_span[1]-updated_p1_span[0])
		print(f"The -1 span for tracking is:", updated_m1_span, " length: ", updated_m1_span[1]-updated_m1_span[0])
		print(f"The span for origin occupancy is:", updated_origin_span, " length: ", 
			updated_origin_span[1]-updated_origin_span[0])

		# Track the +1 nucleosome position. Tracker selects the nucleosome positions
		# of the +1 search range
		def create_tracker(genomic_span, frag_lens, window, tracker_type='nuc_movement'):
			tracker = ChromatinMetricTracking(chrom_model=self)
			tracker.select_range(genomic_span, frag_lens)
			tracker.find_peak_and_update_genomic_positions(window=window)
			if tracker_type == 'nuc_movement': tracker.track_genomic_movement()
			elif tracker_type == 'occupancy': tracker.track_occupancy()
			else: raise ValueError("Unknown parameter: ", tracker_type)
			return tracker

		self.p1_tracker = create_tracker(updated_p1_span, nuc_lens, window=198)
		self.m1_tracker = create_tracker(updated_m1_span, nuc_lens, window=198)
		self.origin_tracker = create_tracker(updated_origin_span, origin_frag_lens, 
			window=198, tracker_type='occupancy')


	def plot_nfr_origin_occ_comparision(self, t_tps=None):
		from src.helpers import normalize_max_min

		t_indices = self.config.get_Hpositions_for_branch('t')

		if t_tps is None:
			t_tps = self.config.get_timepoints_for_branch('t')

		# Compute the NFR size per time
		nfr_size = self.p1_tracker.called_peak_weighted_mean -\
			self.m1_tracker.called_peak_weighted_mean

		# Get the top branch values for NFR length and origin occupancy
		origin_occ = self.origin_tracker.total_occupancy[t_indices]
		nfr_size_t = nfr_size[t_indices]

		normalized_origin_occ_t = normalize_max_min(origin_occ.values)
		normalized_nfr_size_t = normalize_max_min(nfr_size_t.values)

		fig = plt.figure(figsize=(12, 3))

		def plot_comparison(origin_occ, nfr_size_t):
			cmap = plt.get_cmap('Spectral')
			plt.plot(t_tps, origin_occ, label="Origin occupancy", color=cmap(0.9))
			plt.plot(t_tps, nfr_size_t, label="NFR length", color=cmap(0.1))
			plt.legend()

			ax = plt.gca()
			draw_phase_label_annotations(ax, self.config, flip=True, annotations_x=-0.13)
			plt.xlim(t_tps[0], t_tps[-1])
			plt.xticks([])
			plt.yticks([])

		plt.subplot(1, 2, 1)
		plot_comparison(origin_occ.values-origin_occ.values.min() + 20, 
			nfr_size_t.values - nfr_size_t.values.min() + 20)
		plt.ylabel("Occupancy and length")
		plt.ylim(-15, 200)

		plt.subplot(1, 2, 2)
		plot_comparison(normalized_origin_occ_t, normalized_nfr_size_t)
		plt.ylabel("Normalized occupancy and length")
		plt.ylim(-0.25, 1.6)

		if self.origin.strand == '-':
			# flip the xlims
			xlim = plt.xlims()
			plt.xlim(xlim[1], xlim[0])

		return fig


	def plot_nfr_shift_origin_occupancy(self):
		"""Show the +1 nucleosome shifts"""

		# Show that we can track the +1 nucleosome shift per each phase on a high resolution 
		# timescale

		t_indices = self.config.get_Hpositions_for_branch('t')
		s_indices = self.config.get_Hpositions_for_phase('S')
		start_of_s = s_indices[0]
		t_tps = self.config.get_timepoints_for_branch('t')

		plus_position = self.p1_tracker.called_peak_weighted_mean
		plus_position_movement = plus_position - self.center_origin

		minus_position = self.m1_tracker.called_peak_weighted_mean
		minus_position_movement = minus_position - self.center_origin

		m = len(plus_position)

		fig = plt.figure(figsize=(5, 3))

		ax = plt.gca()

		m1_movement = minus_position_movement[t_indices]
		ax.plot(m1_movement, t_tps, lw=4, color='#555')

		p1_movement = plus_position_movement[t_indices]
		ax.plot(p1_movement, t_tps, lw=4, color='#555')

		ylim = t_tps[0], t_tps[-1]
		ax.set_ylim(ylim)

		# Specific xlims for selected genes
		if self.origin.ars_name == 'ARS1212.5':
			translation = -50
			draw_phase_label_annotations(ax, self.config, annotations_x=-120+translation)
			ax.set_xlim(-133+translation, 250+translation)
		else:
			draw_phase_label_annotations(ax, self.config, annotations_x=-120)
			ax.set_xlim(-133, 250)

		from src.plot_helpers import hide_spines

		# Convert H index to t indices for plotting replication location
		def get_t_tp_from_H_index(config, H_index):
			t_indices = config.get_Hpositions_for_branch('t')
			t_tps = config.get_timepoints_for_branch('t')
			index_t = np.where(t_indices == H_index)[0][0]
			tp = t_tps[index_t]
			return tp

		# Plot the replication time
		repl_tp = get_t_tp_from_H_index(self.config, self.origin.replication_index)
		plt.axhline(repl_tp, c='black', ls='dotted', lw=0.5, zorder=0)


		# Plot the boundaries of S
		s_start_tp = get_t_tp_from_H_index(self.config, s_indices[0])
		plt.axhline(s_start_tp, c='black', ls='solid', lw=0.5, zorder=0)
		s_end_tp = get_t_tp_from_H_index(self.config, s_indices[-1])
		plt.axhline(s_end_tp, c='black', ls='solid', lw=0.5, zorder=0)

		ax.set_yticks([])
		plt.suptitle(f"{self.origin.ars_name}", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.8)

		origin_occupancy = self.origin_tracker.total_occupancy

		color_values = origin_occupancy[t_indices].values

		# Plot the origin occupancy as a colormap scatter plot on the 
		# center of the tracked origin location
		x = self.origin_tracker.get_center_selected_bp()
		origin_pos = x - self.center_origin
		origin_pos_extent = origin_pos-10, origin_pos+10
		plt.imshow(color_values.reshape((-1, 1)), extent=[origin_pos_extent[0], origin_pos_extent[1], 
			t_tps[0], t_tps[-1]], cmap='Oranges', aspect='auto', origin='lower', interpolation='none')
		plt.axvline(origin_pos_extent[0], c='#666', lw=0.75,)
		plt.axvline(origin_pos_extent[1], c='#666', lw=0.75,)

		return fig

	def plot_origin_trackers(self):

		fig, ax = self.p1_tracker.plot_selected_region()
		self.m1_tracker.plot_selected_range_rect(ax)
		self.origin_tracker.plot_selected_range_rect(ax)
		plt.title(f"{self.origin.ars_name}\nOrigin nucleosome and subnucleosome tracking regions")
		return fig


	def get_origin_tracking_df(self):
		p1 = self.p1_tracker.called_peak_weighted_mean
		m1 = self.m1_tracker.called_peak_weighted_mean
		origin_occupancy = self.origin_tracker.total_occupancy

		df = pd.DataFrame({
			'+1': p1, '-1': m1, 'origin_occupancy': origin_occupancy
		})
		return df