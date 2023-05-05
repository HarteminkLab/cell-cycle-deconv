function [model] = deconvolve(model)

	WAVETYPE = "Symmlet";
	WAVEPAR = 5;

	g = model.g;
	mean_g = mean(g);
	H = model.H;
	gamma = model.gm;
	Hsize = size(H, 2);

	% Create the fs that we will want to enforce smoothing against
	% they will need to be padded if necessary to be a power of two
	% if so, we will use the end of the H matrix as padding
	[f_initial, f_top, f_bottom, ...
	 f_initial_list, f_top_list, f_bottom_list, ...
	 padding] = createFs(model);

	% Construct the Wavelets
	W1 = getWaveletKernel(WAVETYPE, length(f_initial), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_bottom), WAVEPAR);

	% Scale weighting between W2 and W1 smoothness
	% Time in R, G1, postG1 is roughly 1.5 times as long as DG1 + postG1 (in the original data set)
	% Trying 1.0 in the Yulong Cell Cycle Dataset as R is much shorter
	w = 1.15;

	cvx_begin
		cvx_quiet(true);

		% The f variable is the width of H with any additional padding
		% needed so that the smoothing wavelet function operates on a power of 2
		variable f(Hsize+padding);

		% The fit error, get the relevant indices of f
		% Skipping the first padding indices and removing the last padding indices
		minimize(...
			square_pos(norm(H*f(1:Hsize)./g'-1, 2)) ... % residual norm: the fit error
			+ gamma*(norm(W1*f(f_initial), 1) + ...     % W1, solution norm: a measure of 
					 w*norm(W2*f(f_bottom), 1) ...      % W2, the smoothness/complexity of the solution
					 )/mean_g ...
		);

		subject to
			f>=0;
	cvx_end

	f_final = f(1:end-padding);
	model.f = f_final;
	model.f_initial = f_initial;
	model.f_top = f_top;
	model.f_bottom = f_bottom;
	model.f_initial_list = f_initial_list;
	model.f_top_list = f_top_list;
	model.f_bottom_list = f_bottom_list;

	g_avg = mean(model.g);

	if strcmp(model.datatype, Deconv.DECONV_JOINT)
		glen = length(model.g);
		pred_g = model.H*model.f;
		spl_idx = size(model.timepoints1, 2);
		pred_g1 = pred_g(1:spl_idx);
		pred_g2 = pred_g(spl_idx+1:end);
		model.pred_g1 = pred_g1;
		model.pred_g2 = pred_g2;
	else
		pred_g = model.H*model.f;
	end

	model.pred_g = pred_g;

	rn = square_pos(norm(H*f_final./g'-1, 2));

	% Compute the solution norm
	% The worry is that f_initial and f_bottom may be indexing the padded f vector not the
	% solution f vector
	sn = (norm(W1*f([f_initial]), 1) + ...
		  norm(W2*f([f_bottom]), 1))/mean_g;

	model.rn = rn;
	model.sn = sn;

end
