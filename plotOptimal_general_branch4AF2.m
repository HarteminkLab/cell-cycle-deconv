function [model] = plotOptimal_general_branch4AF(model, fig_flag, imagedir)

%close all;

log_space = 0;

global SLIENCE;

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;
global DECONV_BETAPOS;

global DECONV_JOINT;
global DECONV_DATA1;
global DECONV_DATA2;

mu0 = model.lengths(DECONV_MU0POS);
lambda = model.lengths(DECONV_LAMBDAPOS);
alpha = model.alpha;
beta = model.lengths(DECONV_BETAPOS);

f = model.f;
orfname = model.orig_orfname;

logspace = 0;
fit_fig_flag = 1;

% get intervals and labels from model structure

t_x = [];
t_y = [];
i_x = [];
i_y = [];

border = 40;
if strcmp(model.datatype, DECONV_JOINT)
	xlim_range = max(max(model.timepoints{1}), max(model.timepoints{2})) + border;
else
	xlim_range = max(model.timepoints{1}) + border;
end


% i_y = RC
for i = 1:length(model.i_intervals)
	i_names{1} = model.i_intervals{i}{1};
	idx = model.i_intervals{i}{2};
	len = size(model.Hsegments{idx}, 2);
	start = 0;
	for b = 1:idx-1
		start = start+size(model.Hsegments{b}, 2);
	end
	i_y = [i_y f([(start+1):1:(start+len)])'];
end

% t_y = C
for i = 1:length(model.t_intervals)
	idx = model.t_intervals{i}{2};
	len = size(model.Hsegments{idx}, 2);
	start = 0;
	for b = 1:idx-1
		start = start+size(model.Hsegments{b}, 2);
	end
	t_y = [t_y f([(start+1):1:(start+len)])'];
end

% i_x = [0, mu0+lambda-alpha]
i_segments = {};
offset_i_min = inf;
%offset_i_max = -inf;
for i = 1:length(model.iList)
	offset_i_min = min(offset_i_min, min(model.iList{i}'));
%	offset_i_max = max(offset_i_max, max(model.iList{i}'));
end
for i = 1:length(model.iList)
	list = model.iList{i}';
	i_x = [i_x; list(1:length(list)-1)-offset_i_min];
	i_segments{i}{1} = list(1)-offset_i_min;
	i_segments{i}{2} = list(end-1)-offset_i_min;
	i_segments{i}{3} = model.i_intervals{i}{1};
end

% t_x = [0 lambda]
t_segments = {};
offset_t_min = inf;
for i = 1:length(model.tList)
	offset_t_min = min(offset_t_min, min(model.tList{i}'));
end
for i = 1:length(model.tList)
	list = model.tList{i}';
	t_x = [t_x; list(1:length(list)-1)-offset_t_min];
	t_segments{i}{1} = list(1)-offset_t_min;
	t_segments{i}{2} = list(end-1)-offset_t_min;
	t_segments{i}{3} = model.t_intervals{i}{1};
end

avg_alpha = mean(alpha);

% -------------------
C_g1 = lambda * beta;
C_s = 0.2 * lambda;
% -------------------
xticks_init = [0];
xticks_label = {'|'};
for i = 1:length(i_segments)

	if regexpi(i_segments{i}{3}, '^C')
		s = i_segments{i}{1};
		e = i_segments{i}{2};
		C_len = e - s;
		C_g2m = C_len-C_g1-C_s;

		xticks_init = [xticks_init s+C_g1/2 s+C_g1 ...
		s+C_g1+C_s/2 s+C_g1+C_s ...
		s+C_g1+C_s+C_g2m/2 e];

		xticks_label = {xticks_label{:} 'G1' '|' 'S' '|' 'G2/M' '|'};
	else
		mid = (i_segments{i}{1}+i_segments{i}{2})/2;
		xticks_init = [xticks_init mid i_segments{i}{2}];
		xticks_label{2*i} = i_segments{i}{3};
		xticks_label{2*i+1} = '|';
	end
end

top_offset = xticks_init(end);
xticks_top = [];
for i = 1:length(t_segments)
	if regexpi(t_segments{i}{3}, '^C')
		s = t_segments{i}{1};
		e = t_segments{i}{2};
		C_len = e - s;
		C_g2m = C_len-C_g1-C_s;

		xticks_top = [xticks_top s+C_g1/2 s+C_g1 ...
		s+C_g1+C_s/2 s+C_g1+C_s ...
		s+C_g1+C_s+C_g2m/2 e]+top_offset;

		xticks_label = {xticks_label{:} 'G1' '|' 'S' '|' 'G2/M' '|'};
	else
		mid = (t_segments{i}{1}+t_segments{i}{2})/2;
		xticks_top = [xticks_top mid t_segments{i}{2}]+top_offset;
		xticks_label = {xticks_label{:} t_segments{i}{3} '|'};
	end
end
xticks = [xticks_init xticks_top];

% -------------------
mt = regexprep(model.modeltype, '_', '-');

data_label = 'alpha factor';

if fig_flag == 1
	% plot fit figure
	if fit_fig_flag
		fig1 = figure;
		set(fig1, 'OuterPosition', [1 1 800 600]);
		if ~SLIENCE
			disp('Plotting fit figure');
		end

		names = {};
		intervals = {};

		% JOINT
		if strcmp(model.datatype, DECONV_JOINT)
			glen = length(model.g);

			pred_g = model.H * model.f;
			tm1_len = length(model.timepoints{1});
			g1 = model.g(1:tm1_len);
			g2 = model.g(tm1_len+1:end);
			pred_g1 = pred_g(1:tm1_len);
			pred_g2 = pred_g(tm1_len+1:end);
			R2_wt1 = rsquare(g1, pred_g1);
			R2_wt2 = rsquare(g2, pred_g2);
			model.R2 = [R2_wt1 R2_wt2];

			tm1 = model.timepoints{1};
			tm2 = model.timepoints{2};

			subplot(2,1,1);

			if ~log_space
				plot(tm1, pred_g1, '-o', 'LineWidth', 3, 'color', 'g');
			else
				semilogy(tm1, pred_g1, '-o', 'LineWidth', 3, 'color', 'g');
			end
			hold on;
			if ~log_space
				plot(tm1, g1, '-o', 'LineWidth', 3, 'color', 'r');
			else
				semilogy(tm1, g1, '-o', 'LineWidth', 3, 'color', 'r');
			end
			titlename=sprintf('%s (%s; rep1; %s; %d; %0.3g; R^2=%f)', orfname, data_label, mt, alpha(1), model.gm, model.R2(1));
			title(titlename);

			ylim([0 max([max(pred_g1) max(g1)])*1.2]);
			xlim([0 xlim_range]);
			legend(sprintf('deconv fit of %s', orfname), sprintf('%s (rep 1)', data_label), 'Location', 'Best');
			box off;

			subplot(2,1,2);
			hold on;

			if ~log_space
				plot(tm2, pred_g2, '-o', 'LineWidth', 3, 'color', 'g');
			else
				semilogy(tm2, pred_g2, '-o', 'LineWidth', 3, 'color', 'g');
			end
			hold on;
			if ~log_space
				plot(tm2, g2, '-o', 'LineWidth', 3, 'color', 'r');
			else
				semilogy(tm2, g2, '-o', 'LineWidth', 3, 'color', 'r');
			end

			titlename=sprintf('%s (%s; rep2; %s; %d; %0.3g; R^2=%f)', orfname, data_label, mt, alpha(2), model.gm, model.R2(2));
			title(titlename);

			ylim([0 max([max(pred_g2) max(g2)])*1.2]);
			xlim([0 xlim_range]);
			legend(sprintf('deconv fit of %s', orfname), sprintf('%s (rep 2)', data_label), 'Location', 'Best');
			box off;
		else
			g = model.g;

			pred_g = model.H*model.f;
			R2 = rsquare(model.g, pred_g);
			model.R2 = [R2];
			plot(model.timepoints{1}, pred_g, '-o', 'LineWidth', 3, 'color', 'b');
			hold on;
			plot(model.timepoints{1}, g, '-o', 'LineWidth', 3, 'color', 'r');

			titlename=sprintf('%s (%s; %s; %d; %0.3g; R^2=%f)', orfname, model.datatype, mt, alpha(1), model.gm, model.R2(1));
			title(titlename);

			ylim([0 max([max(pred_g) max(g)])*1.2]);
			xlim([0 300]);
			legend(names{:}, sprintf('%s', model.datatype), 'Location', 'Best');
			box off;
		end

		if nargin == 3
			figname = strcat(imagedir, '/', model.orfname, '-fit.png');
%			set(fig1, 'Units', 'pixels');
%			set(fig1, 'papersize', [4.25 5.5]);
%			set(fig1, 'PaperPositionMode', 'auto');
%			print(fig1, '-dpng', '-r100', figname); 
			saveas(gcf, strcat(imagedir, '/', model.orfname, '-fit.png'), 'png');
			close;
		end
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	y_lim = max([max(t_y), max(i_y)])*1.2; 
	areacolor = [0 98 139]./255;

	if strcmp(DECONV_JOINT, model.datatype)
		titlename = sprintf('%s ', orfname);
	else
		titlename = sprintf('%s ', orfname);
	end
   

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	fig2 = figure;
	set(fig2, 'OuterPosition', [800 600 800 400]);

	idx = 1;
	for i = 1:length(model.iList)
		list = model.iList{i}';
		halfbin = (list(2)-list(1))/2;
		len = length(list)-1;
		bar(i_x(idx:idx+len-1)'+halfbin, i_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
		idx = idx+len;
		hold on;
	end
	idx = 1;
	for i = 1:length(model.tList)
		list = model.tList{i}';
		len = length(list)-1;
		halfbin = (list(2)-list(1))/2;
		bar(t_x(idx:idx+len-1)'+halfbin+top_offset, t_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
		idx = idx+len;
		hold on;
	end
	hold off;

	set(gca, 'XTick', xticks);
	set(gca, 'XTickLabel', xticks_label);
	xlim([0 top_offset+t_x(end)]);
	ylim([0 y_lim]);
	if strcmp(DECONV_JOINT, model.datatype)
		titlename = sprintf('%s (%s; %s; [%d %d]; %0.3g)', orfname, model.datatype, mt, alpha(1), alpha(2), model.gm);
	else
		titlename = sprintf('%s (%s; %s; %d; %0.3g)', orfname, model.datatype, mt, alpha(1), model.gm);
	end
	title(strcat(titlename));

	%%%%%%%%%%%%%%%

  if nargin == 3
%		figname = strcat(imagedir, '/', model.orfname);
%		print(fig2, '-dpng', '-r100', figname); 
		saveas(gcf, strcat(imagedir, '/', model.orfname), 'png');
		close;
	end
end

return;
