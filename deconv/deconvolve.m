function [model] = deconvolve(model)

	WAVETYPE = "Symmlet";
	WAVEPAR = 5;

	g = model.g';

	mean_g = mean(g);
	H = model.H;
	gamma = model.gm;
	Hsize = size(H, 2);

	f_it = [];
	f_i = [];
	i_intervals = model.intervals.initialPhaseMapping;
	i_list = model.intervals.initialTimepointsList;

	t_intervals = model.intervals.topPhaseMapping;
	t_list = model.intervals.topTimepointsList;

	b_intervals = model.intervals.bottomPhaseMapping;
	b_list = model.intervals.bottomTimepointsList;

	[f_i, f_t, f_b, f_initial_list, f_top_list, f_bottom_list] = createFs(model);

	f_it = [f_i f_t];

	f_final = zeros(Hsize,1);

	% =============== From deconv.v2 ===============

	f_b = f_b;
	f_it = f_it;
	factor_fb = 1.5;

	W1 = getWaveletKernel(WAVETYPE, length(f_it), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);

	cvx_begin
		cvx_quiet(true);

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f ./ g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_it),1) + ...
			factor_fb*norm(W2*f(f_b),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_b) = f(f_b);
	f_final(f_it) = f(f_it);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	f = f_final;

	sn = ( norm(W1*f([f_it]),1) + norm(W2*f([f_b]),1) )/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

	pred_g = H*f;

	model.f = f;
	model.sn = sn;
	model.rn = rn;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	model.f_initial = f_i;
	model.f_top = f_t;
	model.f_bottom = f_b;

	model.f_initial_list = f_initial_list;
	model.f_top_list = f_top_list;
	model.f_bottom_list = f_bottom_list;


	if strcmp(model.datatype, Deconv.DECONV_JOINT)
		glen = length(model.g);
		pred_g = model.H*model.f;
		g1 = model.g(1:glen/2);
		g2 = model.g(glen/2+1:glen);
		pred_g1 = pred_g(1:glen/2);
		pred_g2 = pred_g(glen/2+1:glen);
	else
		pred_g = model.H*model.f;
	end

	model.pred_g = pred_g;
    % model.W1 = W1;
    % model.W2 = W2;
    % model.mean_g = mean_g;

    % % Store the smoothing norm
	% model.sn = gamma*(norm(W1*f(f_initial),1) + ...
    %             norm(W2*f(f_top), 1) + ...
    %             norm(W3*f(f_bottom), 1))/mean_g;

    % % The old calculation of the smoothing norm without gamma, for
    % % reference
	% model.sn_nogamma = (norm(W1*f(f_initial),1) + ...
    %                     norm(W2*f(f_top), 1) + ...
    %                     norm(W3*f(f_bottom), 1))/mean_g;

    % % Store the residual norm
    % model.rn = square_pos(norm(model.H*f(1:Hsize) ./ model.g-1, 2));

    % model.err = sqrt(model.rn/length(model.g));
end
