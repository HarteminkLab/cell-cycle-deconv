function [model] = deconvolve(model)

	WAVETYPE = "Symmlet";
	WAVEPAR = 5;
	PADDINGSIZE = 42;

	g = model.g;
	mean_g = mean(g);
	H = model.H;
	gamma = model.gm;

	g = model.g;
	mean_g = mean(g);
	H = model.H;
	gamma = model.gm;
	Hsize = size(H, 2);

	% Unused in optimzation. For recording the top
	% indices for plotting
	f_t = [];
	f_t_list = {};
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		indices = se(1):1:se(2);
		f_t = [f_t indices];
		f_t_list{i} = indices;
	end

	% Construct the f index vector (f_i) for the first
	% Wavelet smoothing criteria W1
	% Initial with padding to make it a power of 2
	f_i = [];
	last = Hsize;
	f_i_front = last+[1:1:PADDINGSIZE];
	last = last+PADDINGSIZE;
	f_i_after = last+[1:1:PADDINGSIZE];
	last = last+PADDINGSIZE;
	f_i_list = {};
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		indices = se(1):1:se(2);
		f_i = [f_i indices];
		f_i_list{i} = indices;

	end

	f_i_pad = [f_i_front f_i f_i_after];

	% Construct the f index vector (f_b) for the second
	% Wavelet smoothing criteria W2
	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		indices = se(1):1:se(2);
		f_b = [f_b indices];
		f_b_list{i} = indices;

	end

	% Construct the Wavelets
	Hsize = size(H, 2);
	W1 = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);


	fixed_f_i_pad = [1:130 217:Hsize+PADDINGSIZE*2];
	fixed_f_b = [131:258];

	model.fixed_f_i_pad = fixed_f_i_pad;
	model.f_i_pad = f_i_pad;

	% Enforce smoothness of the entire padded array
	W = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);

	cvx_begin
		cvx_quiet(true);

		variable f(Hsize+PADDINGSIZE*2); % 258 + 42*2 = 342

		% The fit error, get the relevant indices of f
		% Skipping the first PADDINGSIZE indices and removing the last PADDINGSIZE indices
		minimize(...
			square_pos(norm(H*f(1:Hsize)./g'-1, 2)) ... % fit errors
			+ gamma*(norm(W1*f(fixed_f_i_pad),1) + ...
					 norm(W2*f(fixed_f_b),1) ...
					 )/mean_g ...
		);

		subject to
			f>=0;
	cvx_end

	f_final = f(1:end-PADDINGSIZE*2);
	model.f = f_final;

	model.f_i = f_i;
	model.f_t = f_t;
	model.f_b = f_b;

	model.f_b_list = f_b_list;
	model.f_t_list = f_t_list;
	model.f_i_list = f_i_list;

	g_avg = mean(model.g);

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
end
