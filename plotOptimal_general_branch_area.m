function [model] = plotOptimal_general_branch_area(model, fig_flag, imagedir)

log_space = 1;

global SLIENCE;

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_DELTAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;
global DECONV_BETAPOS;

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

use_standard = 0;

if use_standard
	mu0 = (94.387+101.904)/2;
	lambda = (79.487+82.014)/2;
	delta = (44.318+37.436)/2;
	beta = (0.153+0.165)/2;
	alpha = [26 27];
else
	mu0 = model.lengths(DECONV_MU0POS);
	lambda = model.lengths(DECONV_LAMBDAPOS);
	delta = model.lengths(DECONV_DELTAPOS);
	alpha = model.alpha;
	beta = model.lengths(DECONV_BETAPOS);
end

f = model.f;
orfname = model.orig_orfname;

% get intervals and labels from model structure
t_x = [];
t_y = [];
b_x = [];
b_y = [];
i_x = [];
i_y = [];

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

% b_y = DC
for i = 1:length(model.b_intervals)
	idx = model.b_intervals{i}{2};
	len = size(model.Hsegments{idx}, 2);
	start = 0;
	for b = 1:idx-1
		start = start+size(model.Hsegments{b}, 2);
	end
	b_y = [b_y f([(start+1):1:(start+len)])'];
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

% b_x = [0 delta+lambda]
offset_b_min = inf;
b_segments = {};
for i = 1:length(model.bList)
	offset_b_min = min(offset_b_min, min(model.bList{i}'));
end
for i = 1:length(model.bList)
	list = model.bList{i}';
	b_x = [b_x; list(1:length(list)-1)-offset_b_min];
	b_segments{i}{1} = list(1)-offset_b_min;
	b_segments{i}{2} = list(end-1)-offset_b_min;
	b_segments{i}{3} = model.b_intervals{i}{1};
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

model.t_x = t_x;
model.t_y = t_y;
model.b_x = b_x;
model.b_y = b_y;
model.i_x = i_x;
model.i_y = i_y;

logspace = 0;
fit_fig_flag = 1;

S_phase = 0.2;
if strcmp(model.datatype, DECONV_JOINT)
	avg_alpha = alpha(2);
else
	avg_alpha = alpha(1);
end

s_interval = S_phase*lambda;

% -------------------
xticks_init = [0];
xticks_label_init = {'|'};
for i = 1:length(i_segments)
	name = i_segments{i}{3};
	if strcmp(name, 'postG1')
		smid = i_segments{i}{1} + s_interval/2;
		xticks_init = [xticks_init smid i_segments{i}{1}+s_interval];

		g2mmid = (i_segments{i}{1}+s_interval+i_segments{i}{2})/2;
		xticks_init = [xticks_init g2mmid i_segments{i}{2}];
		xticks_label_init = {xticks_label_init{:}, 'S', '|', 'G2/M', '|'};
		i=i+1;
	elseif strcmp(name, 'CG1')
		mid = (i_segments{i}{1}+i_segments{i}{2})/2;
		xticks_init = [xticks_init mid i_segments{i}{2}];
		xticks_label_init = {xticks_label_init{:}, 'G1', '|'};
	else
		mid = (i_segments{i}{1}+i_segments{i}{2})/2;
		xticks_init = [xticks_init mid i_segments{i}{2}];
		xticks_label_init = {xticks_label_init{:}, i_segments{i}{3}, '|'};
	end
end
%xticks_init(end) = model.i_x(end);

xticks_bottom = [0];
xticks_label_bottom = {'|'};
for i = 1:length(b_segments)
	name = b_segments{i}{3};
	if strcmp(name, 'postG1')
		smid = b_segments{i}{1} + s_interval/2;
		xticks_bottom = [xticks_bottom smid b_segments{i}{1}+s_interval];

		g2mmid = (b_segments{i}{1}+s_interval+b_segments{i}{2})/2;
		xticks_bottom = [xticks_bottom g2mmid b_segments{i}{2}];
		xticks_label_bottom = {xticks_label_bottom{:}, 'S', '|', 'G2/M', '|'};
		i=i+1;
	else
		mid = (b_segments{i}{1}+b_segments{i}{2})/2;
		xticks_bottom = [xticks_bottom mid b_segments{i}{2}];
		xticks_label_bottom = {xticks_label_bottom{:}, b_segments{i}{3}, '|'};
	end
end
%xticks_bottom(end) = model.b_x(end);

xticks_top = [0];
xticks_label_top = {''};
for i = 1:length(t_segments)
	name = t_segments{i}{3};
	if strcmp(name, 'postG1')
		smid = t_segments{i}{1} + s_interval/2;
		xticks_top = [xticks_top smid t_segments{i}{1}+s_interval];

		g2mmid = (t_segments{i}{1}+s_interval+t_segments{i}{2})/2;
		xticks_top = [xticks_top g2mmid t_segments{i}{2}];
		xticks_label_top = {xticks_label_top{:}, 'S', '|', 'G2/M', '|'};
		i=i+1;
	elseif strcmp(name, 'CG1')
		mid = (t_segments{i}{1}+t_segments{i}{2})/2;
		xticks_top = [xticks_top mid t_segments{i}{2}];
		xticks_label_top = {xticks_label_top{:}, 'G1', '|'};
	else
		mid = (t_segments{i}{1}+t_segments{i}{2})/2;
		xticks_top = [xticks_top mid t_segments{i}{2}];
		xticks_label_top = {xticks_label_top{:}, t_segments{i}{3}, '|'};
	end
end
%xticks_top(end) = model.t_x(end);

% -------------------
mt = regexprep(model.modeltype, '_', '-');

if fig_flag == 1
	% plot fit figure
	if fit_fig_flag
		fig1 = figure;
		set(fig1, 'OuterPosition', [1 1 800 600]);
%		if ~SLIENCE
%			disp('Plotting fit figure');
%		end

		if strcmp(model.datatype, DECONV_JOINT)
			glen = length(model.g);
			pred_g = model.H * model.f;
			g1 = model.g(1:glen/2);
			g2 = model.g(glen/2+1:glen);
			pred_g1 = pred_g(1:glen/2);
			pred_g2 = pred_g(glen/2+1:glen);

			R2_wt1 = rsquare(g1, pred_g1);
			R2_wt2 = rsquare(g2, pred_g2);
			
			tm1 = model.timepoints(1,:);
			tm2 = model.timepoints(2,:);

			subplot(2,1,1);
			my_gray = [233 233 233]./255;
			if logspace
				semilogy(tm1, g1, '-o', 'LineWidth', 3, 'color', 'r', 'MarkerFaceColor', my_gray, 'MarkerSize', 6);
			else
				plot(tm1, g1, '-o', 'LineWidth', 3, 'color', 'r', 'MarkerFaceColor', my_gray, 'MarkerSize', 6);
			end
			hold on;
			if logspace
				semilogy(tm1, pred_g1, '-o', 'LineWidth', 2, 'color', 'g', 'MarkerFaceColor', my_gray, 'Markersize', 6);
			else
				plot(tm1, pred_g1, '-o', 'LineWidth', 2, 'color', 'g', 'MarkerFaceColor', my_gray, 'Markersize', 6);
			end

			xlabel('Time (min)');
			ylabel('Expression level (log)');

%			titlename=sprintf('%s (WT1; %s; %d; %0.3g; R^2=%0.3g)', orfname, mt, alpha(1), model.gm, R2_wt1);
			titlename = sprintf('%s (wild-type replication 1, R^2=%0.3f)', orfname, R2_wt1);
			title(titlename);

			ylim([0 max([pred_g1' g1'])*1.2]);
			legend('wild-type 1', 'deconv fitting', 'Location', 'EastOutside');

			subplot(2,1,2);
			if logspace
				semilogy(tm2, g2, '-o', 'LineWidth', 3, 'color', 'r', 'MarkerFaceColor', my_gray, 'Markersize', 6);
			else
				plot(tm2, g2, '-o', 'LineWidth', 3, 'color', 'r', 'MarkerFaceColor', my_gray, 'Markersize', 6);
			end
			hold on;
			if logspace
				semilogy(tm2, pred_g2, '-o', 'LineWidth', 2, 'color', 'g', 'MarkerFaceColor', my_gray, 'Markersize', 6);
			else
				plot(tm2, pred_g2, '-o', 'LineWidth', 2, 'color', 'g', 'MarkerFaceColor', my_gray, 'Markersize', 6);
			end

			xlabel('Time (min)');
			ylabel('Expression level (log)');
%			titlename=sprintf('%s (WT2; %s; %d; %0.3g; R^2=%0.3g)', orfname, mt, alpha(2), model.gm, R2_wt2);
			titlename = sprintf('%s (wild-type replication 2, R^2=%0.3f)', orfname, R2_wt2);
			title(titlename);

			ylim([0 max([pred_g2' g2'])*1.2]);
			legend('wild-type 2', 'deconv fitting', 'Location', 'EastOutside');
		else
			g = model.g;
			pred_g = model.H * model.f;

			R2 = rsquare(g, pred_g);

			plot(model.timepoints, g, '-o', 'LineWidth', 3, 'color', 'r');
			hold on;
			plot(model.timepoints, pred_g, '-o', 'LineWidth', 2, 'color', 'g');

			xlabel('Time (min)');
			ylabel('Expression level (log-scale)');
			titlename=sprintf('%s (%s; %s; %d; %0.3g; R^2=%0.3g)', orfname, model.datatype, mt, alpha(1), model.gm, R2);
			title(titlename);

			ylim([0 max([pred_g' g'])*1.2]);
			if strcmp(model.datatype, DECONV_WT1);
				legend('wild-type 1', 'deconv fitting', 'Location', 'EastOutside');
			else
				legend('wild-type 2', 'deconv fitting', 'Location', 'EastOutside');
			end

		end

		if nargin == 3
			saveas(gcf, strcat(imagedir, '/', model.orfname, '-fit.png'), 'png');
			close;
		end
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	y_lim = max([max(b_y), max(t_y), max(i_y)])*1.2; 
	areacolor = [0 98 139]./255;

 	if DECONV_KERNEL == DECONV_DIFF
		kernel = 'DIFF';
	elseif DECONV_KERNEL == DECONV_WAVELET
		kernel = 'WAVELET';
	end
   
	if strcmp(DECONV_JOINT, model.datatype)
		titlename = sprintf('%s ', orfname);
	else
		titlename = sprintf('%s ', orfname);
	end
    
	fig2 = figure;
	set(fig2, 'OuterPosition', [400 600 1200 600]);
	heigth = 0.35;
	w = 0.88;
	mid_interval = 0.0006;
	maxw = (lambda+mu0-avg_alpha+lambda+delta)*w/(w-mid_interval);
%	start_left = (1-w)/2;
	left_mar = 0.03;
	start_left = 1-w-left_mar;
	end_left = start_left+w;
	top_base = 0.52;
	off_base = 0.025;
	bottom_base = 0.05;

	% initial branch
	left = start_left;
	curw = w*(lambda+mu0-avg_alpha)/maxw;
	h1 = axes('Position', [left, top_base+off_base, curw, heigth]);
	idx = 1;
	for i = 1:length(model.iList)
		list = model.iList{i}';
		halfbin = (list(2)-list(1))/2;
		len = length(list)-1;
		bar(i_x(idx:idx+len-1)'+halfbin, i_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
%		if i==1
%			plot(i_x(idx:idx+len), i_y(idx:idx+len), 'linewidth', 1, 'color', areacolor);
%		else
%			plot(i_x(idx:idx+len-1), i_y(idx:idx+len-1), 'linewidth', 1, 'color', areacolor);
%		end
		idx = idx+len;
		hold on;
	end

	set(h1, 'XTick', xticks_init);
	set(h1, 'XTickLabel', xticks_label_init);
	xlim([0 lambda+mu0-avg_alpha]);
	ylim([0 y_lim]);
	if strcmp(DECONV_JOINT, model.datatype)
	%	titlename = sprintf('%s (%s; %s; [%d %d]; %0.3g)', orfname, model.datatype, mt, alpha(1), alpha(2), model.gm);
		titlename = sprintf('%s', orfname);
	else
%		titlename = sprintf('%s (%s; %s; %d; %0.3g)', orfname, model.datatype, mt, alpha(1), model.gm);
		titlename = sprintf('%s', orfname);
	end
	title(strcat(titlename, ' (initial)'));
	ylabel('Expression level');

	% top branch
	left = left+curw+mid_interval;
	curw = w/maxw*(lambda);
	h2 = axes('Position', [left, top_base+off_base, curw, heigth]);
	idx = 1;
	for i = 1:length(model.tList)
		list = model.tList{i}';
		len = length(list)-1;
		halfbin = (list(2)-list(1))/2;
		bar(t_x(idx:idx+len-1)'+halfbin, t_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
%		if i==1
%			plot(t_x(idx:idx+len-1), t_y(idx:idx+len-1), 'linewidth', 1, 'color', areacolor);
%		else
%			plot(t_x(idx-1:idx+len-1), t_y(idx-1:idx+len-1), 'linewidth', 1, 'color', areacolor);
%		end
		idx = idx+len;
		hold on;
	end

  set(h2, 'XTick', xticks_top);
  set(h2, 'XTickLabel', xticks_label_top);
	set(h2, 'Ytick', []);
	set(h2, 'YTickLabel', {});
  xlim([0 lambda]);
  ylim([0 y_lim]);
	title(strcat(titlename, ' (mother)'));
  
	% bottom branch
	left = left;
	curw = w/maxw*(delta+lambda);
	h3 = axes('Position', [left, bottom_base+off_base, curw, heigth]);
	idx = 1;
	for i = 1:length(model.bList)
		list = model.bList{i}';
		len = length(list)-1;
		halfbin = (list(2)-list(1))/2;
		bar(b_x(idx:idx+len-1)'+halfbin, b_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
%		if i==1
%			plot(b_x(idx:idx+len-1), b_y(idx:idx+len-1), 'linewidth', 1, 'color', areacolor);
%		else
%			plot(b_x(idx-1:idx+len-1), b_y(idx-1:idx+len-1), 'linewidth', 1, 'color', areacolor);
%		end
		idx = idx+len;
		hold on;
	end

  set(h3, 'XTick', xticks_bottom);
  set(h3, 'XTickLabel', xticks_label_bottom);
  xlim([0 delta+lambda]);
  ylim([0 y_lim]);
	ylabel('Expression level');
	title(strcat(titlename, ' (daughter)'));
  
  if nargin == 3
		saveas(gcf, strcat(imagedir, '/', model.orfname), 'png');
		close;
	end

end

return;
