function [model] = plotOptimal_general_parts(model, fig_flag, imagename)

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

DRAW_NORMALIZAING = 0;

logspace = 0;
fit_fig_flag = 1;

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
	if DRAW_NORMALIZAING
		gap = model.iList{i}(2) - model.iList{i}(1);
		disp(sprintf('gap in i %f', gap));
	else
		gap = 1;
	end
	i_y = [i_y f([(start+1):1:(start+len)])'./gap];
end

% b_y = DC
for i = 1:length(model.b_intervals)
	idx = model.b_intervals{i}{2};
	len = size(model.Hsegments{idx}, 2);
	start = 0;
	for b = 1:idx-1
		start = start+size(model.Hsegments{b}, 2);
	end
	if DRAW_NORMALIZAING
		gap = model.bList{i}(2) - model.bList{i}(1);
		disp(sprintf('gap in b %f', gap));
	else
		gap = 1;
	end
	b_y = [b_y f([(start+1):1:(start+len)])'./gap];
end

% t_y = C
for i = 1:length(model.t_intervals)
	idx = model.t_intervals{i}{2};
	len = size(model.Hsegments{idx}, 2);
	start = 0;
	for b = 1:idx-1
		start = start+size(model.Hsegments{b}, 2);
	end
	if DRAW_NORMALIZAING
		gap = model.tList{i}(2) - model.tList{i}(1);
		disp(sprintf('gap in t %f', gap));
	else
		gap = 1;
	end
	t_y = [t_y f([(start+1):1:(start+len)])'./gap];
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

S_phase = 0.2;
avg_alpha = mean(alpha);

% -------------------
xticks_init = [0];
xticks_label_init = {'|'};
for i = 1:length(i_segments)
	mid = (i_segments{i}{1}+i_segments{i}{2})/2;
	xticks_init = [xticks_init mid i_segments{i}{2}];
	xticks_label_init{2*i} = i_segments{i}{3};
	xticks_label_init{2*i+1} = '|';
end

xticks_bottom = [0];
xticks_label_bottom = {'|'};
for i = 1:length(b_segments)
	mid = (b_segments{i}{1}+b_segments{i}{2})/2;
	xticks_bottom = [xticks_bottom mid b_segments{i}{2}];
	xticks_label_bottom{2*i} = b_segments{i}{3};
	xticks_label_bottom{2*i+1} = '|';
end

xticks_top = [0];
xticks_label_top = {'|'};
for i = 1:length(t_segments)
	mid = (t_segments{i}{1}+t_segments{i}{2})/2;
	xticks_top = [xticks_top mid t_segments{i}{2}];
	xticks_label_top{2*i} = t_segments{i}{3};
	xticks_label_top{2*i+1} = '|';
end

% -------------------
mt = regexprep(model.modeltype, '_', '-');

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
		for idx = 1:length(model.relations)
			names{idx} = model.relations{idx}{1};
			branch = model.relations{idx}{2};
			branch_idx = model.relations{idx}{3};
			branch_idx = str2num(branch_idx)+1;
			
			if strcmp(branch, 'i')
				seg_idx = model.i_intervals{branch_idx}{2};
			elseif strcmp(branch, 't')
				seg_idx = model.t_intervals{branch_idx}{2};
			elseif strcmp(branch, 'b')
				seg_idx = model.b_intervals{branch_idx}{2};
			end

			seg = model.Hpos{seg_idx};
			% start
			% end
			intervals{idx}{1} = seg(1);
			intervals{idx}{2} = seg(2);
		end

		if strcmp(model.datatype, DECONV_JOINT)
			glen = length(model.g);

			pred_g = [];
			for idx = 1:length(model.relations)
				seg_start = intervals{idx}{1};
				seg_end = intervals{idx}{2};
				curf = zeros(length(model.f),1);
				curf(seg_start:seg_end) = model.f(seg_start:seg_end);
				pred_g = [pred_g model.H*curf];
			end

%			pred_g = model.H * model.f;
			g1 = model.g(1:glen/2);
			g2 = model.g(glen/2+1:glen);
			pred_g1 = pred_g(1:glen/2,:);
			pred_g2 = pred_g(glen/2+1:glen,:);

			tm1 = model.timepoints(1,:);
			tm2 = model.timepoints(2,:);

			subplot(2,1,1);
%			if logspace
%				semilogy(tm1, g1, '-o', 'LineWidth', 3, 'color', 'r');
%			else
%			end
%			if logspace
%				semilogy(tm1, pred_g1, '-o', 'LineWidth', 2, 'color', 'g');
%			else
%			end

			area(tm1, pred_g1);
			colormap summer;
			hold on;
			plot(tm1, g1, '-o', 'LineWidth', 3, 'color', 'r');
			titlename=sprintf('%s (WT1; %s; %d; %0.3g; R^2=%f)', orfname, mt, alpha(1), model.gm, model.R2(1));
			title(titlename);

			ylim([0 max([sum(pred_g1') g1'])*1.2]);
%			ylim([0 max([pred_g1' g1'])*1.2]);
			xlim([0 300]);
			legend(names{:}, 'WT1');

			subplot(2,1,2);
%			if logspace
%				semilogy(tm2, g2, '-o', 'LineWidth', 3, 'color', 'r');
%			else
%			end
			hold on;
%			if logspace
%				semilogy(tm2, pred_g2, '-o', 'LineWidth', 2, 'color', 'g');
%			else
%				plot(tm2, pred_g2, '-o', 'LineWidth', 2, 'color', 'g');
%			end

			area(tm2, pred_g2);
			colormap summer;
			hold on;
			plot(tm2, g2, '-o', 'LineWidth', 3, 'color', 'r');

			titlename=sprintf('%s (WT2; %s; %d; %0.3g; R^2=%f)', orfname, mt, alpha(2), model.gm, model.R2(2));
			title(titlename);

			ylim([0 max([sum(pred_g2') g2'])*1.2]);
	%		ylim([0 max([pred_g2' g2'])*1.2]);
			xlim([0 300]);
			legend(names{:}, 'WT2');
		else
			g = model.g;

			pred_g = [];
			for idx = 1:length(model.relations)
				seg_start = intervals{idx}{1};
				seg_end = intervals{idx}{2};
				curf = zeros(length(model.f),1);
				curf(seg_start:seg_end) = model.f(seg_start:seg_end);
				pred_g = [pred_g model.H*curf];
			end

%			pred_g = model.H * model.f;
%			semilogy(model.timepoints, g, '-o', 'LineWidth', 3, 'color', 'r');
			area(model.timepoints, pred_g);
			hold on;
			plot(model.timepoints, g, '-o', 'LineWidth', 3, 'color', 'r');
%			semilogy(model.timepoints, pred_g, '-o', 'LineWidth', 2, 'color', 'g');

%			titlename=sprintf('%s data fitting (%s, alpha=%d, gamma=%0.5g)', orfname, model.datatype, alpha, model.gm);
			titlename=sprintf('%s (%s; %s; %d; %0.3g; R^2=%f)', orfname, model.datatype, mt, alpha(1), model.gm, model.R2(1));
			title(titlename);

			ylim([0 max([sum(pred_g') g'])*1.2]);
	%		ylim([0 max([pred_g' g'])*1.2]);
			xlim([0 300]);
			legend(names{:}, sprintf('%s', model.datatype));
		end

		if nargin == 3
			saveas(gcf, strcat(imagename, '.fit.png'), 'png');
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
	%	titlename = sprintf('%s (JOINT; %s; [%d,%d]; %0.3g)', orfname, mt, alpha(1), alpha(2), model.gm);
		titlename = sprintf('%s ', orfname);
	else
	%	titlename = sprintf('%s (%s; %s; %d; %0.3g)', orfname, model.datatype, mt, alpha(1), model.gm);
		titlename = sprintf('%s ', orfname);
	end
    
	fig2 = figure;
	set(fig2, 'OuterPosition', [800 600 800 1000]);
	height = 0.22;
	maxw = max([lambda lambda+mu0-avg_alpha lambda+delta]);
	w = 0.8;

	% figure for f_i
	curw = w/maxw*(lambda+mu0-avg_alpha);
	left = 0.9-curw;
	h1 = axes('Position', [left, 0.7+0.025, curw, height]);
	idx = 1;
	for i = 1:length(model.iList)
		list = model.iList{i}';
		halfbin = (list(2)-list(1))/2;
		len = length(list)-1;
		bar(i_x(idx:idx+len-1)'+halfbin, i_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
		idx = idx+len;
		hold on;
	end

	set(h1, 'XTick', xticks_init);
	set(h1, 'XTickLabel', xticks_label_init);
	xlim([0 lambda+mu0-avg_alpha]);
	ylim([0 y_lim]);
	if strcmp(DECONV_JOINT, model.datatype)
		titlename = sprintf('%s (%s; %s; [%d %d]; %0.3g)', orfname, model.datatype, mt, alpha(1), alpha(2), model.gm);
	else
		titlename = sprintf('%s (%s; %s; %d; %0.3g)', orfname, model.datatype, mt, alpha(1), model.gm);
	end
	title(strcat(titlename, ' (I branch)'));

	% figure for f_b
	curw = w/maxw*(delta+lambda);
	left = 0.9-curw;
	h2 = axes('Position', [left, 0.375+0.025, curw, height]);
	idx = 1;
	for i = 1:length(model.bList)
		list = model.bList{i}';
		len = length(list)-1;
		halfbin = (list(2)-list(1))/2;
		bar(b_x(idx:idx+len-1)'+halfbin, b_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
		idx = idx+len;
		hold on;
	end
	
	set(h2, 'XTick', xticks_bottom);
	set(h2, 'XTickLabel', xticks_label_bottom);
	xlim([0 delta+lambda]);
	ylim([0 y_lim]);
	title(strcat(titlename, ' (B branch)'));

	% figure for f_t
	curw = w/maxw*(lambda);
	left = 0.9-curw;
	h3 = axes('Position', [left, 0.05+0.025, curw, height]);
	idx = 1;
	for i = 1:length(model.tList)
		list = model.tList{i}';
		len = length(list)-1;
		halfbin = (list(2)-list(1))/2;
		bar(t_x(idx:idx+len-1)'+halfbin, t_y(idx:idx+len-1), 1, 'FaceColor', areacolor, 'LineWidth', 1, 'EdgeColor', areacolor);
		idx = idx+len;
		hold on;
	end

  set(h3, 'XTick', xticks_top);
  set(h3, 'XTickLabel', xticks_label_top);
  xlim([0 lambda]);
  ylim([0 y_lim]);
	title(strcat(titlename, ' (T branch)'));
    
  if nargin == 3
		saveas(gcf, imagename, 'png');
		close;
	end

end

return;
