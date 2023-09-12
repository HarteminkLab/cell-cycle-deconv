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

    % Wondering what H, f, and g look like:
    fprintf("\n");
    fprintf("------------------------\n");

    fprintf('The size: of H is: ');
    fprintf('%d ', size(H));
    fprintf('\n');
    fprintf("------------------------\n");

    fprintf('Hsize is: ');
    fprintf('%d ', Hsize);
    fprintf('\n');
    fprintf("------------------------\n");

    fprintf('The size: of g is: ');
    fprintf('%d ', size(g));
    fprintf('\n');
    fprintf("------------------------\n");

    f = zeros(Hsize+padding);

    fprintf('The size of f() is: ');
    fprintf('%d ', size(f(1:Hsize)));
    fprintf('\n');
    fprintf("------------------------\n");

    % So H is 30x258
    % g is 30x1
    %     and g' is 1x30
    % and f is 342, but f(1:Hsize) should be 
    %      258x1

    % Thus the norm equation is:
    %
    %      H * f_ ./ g'
    %
    %     [  30 x 258  ]   *   [ 258 x 1 ]    ./      1 x 30

    res = H*f(1:Hsize)';

    fprintf("The result of H*f' is: ");
    fprintf("%d ", size(res));
    fprintf("\n");

    error("TODO: Ending execution early for debugging.");
    
	cvx_begin
		cvx_quiet(true);

		variable f(Hsize+padding);

		% The fit error, get the relevant indices of f
		% Skipping the first padding indices and removing the last padding 
        % indices
		minimize(...
			square_pos(norm(H*f(1:Hsize)'./g'-1, 2)) ... % fit errors
			+ gamma*(norm(W1*f(f_initial),1) + ...
					 norm(W2*f(f_bottom),1) ...
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
		g1 = model.g(1:glen/2);
		g2 = model.g(glen/2+1:glen);
		pred_g1 = pred_g(1:glen/2);
		pred_g2 = pred_g(glen/2+1:glen);
	else
		pred_g = model.H*model.f;
	end

	model.pred_g = pred_g;
end
