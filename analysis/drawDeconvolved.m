function[model] = drawDeconvolved(model)

i_timepoints = [model.iList{1} model.iList{2}];
i_timepoints = i_timepoints(1:end, 3:end)';
b_timepoints = [model.bList{1} model.bList{2}];
b_timepoints = b_timepoints(1:end, 3:end)';
t_timepoints = [model.tList{1} model.tList{2}];
t_timepoints = t_timepoints(1:end, 3:end)';
i_timepoints = [model.iList{1} model.iList{2} model.iList{3}];
i_timepoints = i_timepoints(1:end, 3:end-1)';

b_data = model.f(model.f_b);
t_data = model.f(model.f_t);
i_data = model.f(model.f_i);

f = figure();
f.Position = [100 100 800 400];
tiled_layout = tiledlayout(2, 3);
tiled_layout.Padding = 'compact';
tiled_layout.TileSpacing = 'compact';
title(tiled_layout, model.genename);

% WT1 Raw and fitted plot
nexttile;
hold on;
plot(model.timepoints(1, :), model.g(1, 1:15), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.timepoints(1, :), model.pred_g(1:15, 1), '-', 'color', [28 200 28]/255., 'LineWidth', 2.5);
ylim([0  max(model.g)*1.1]);
% yticks([]);
ylabel('WT 1');
hold off;

% Initial plot
nexttile;
hold on;
plot(model.iList{1}(3:end), model.f(model.f_i_list{1}(2:end)), '-', 'color', [28 26 228]/255., 'LineWidth', 2.5);
plot(model.iList{2}(2:end), model.f(model.f_i_list{2}), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.iList{3}(3:end), model.f(model.f_i_list{3}(1:end-1)), '-', 'color', [26 228 28]/255., 'LineWidth', 2.5);
% ylim([0  max(model.g)*1.1]);
% yticks([]);
ylabel('Initial');
hold off;

% Mother plot
nexttile;
hold on;
% plot(t_timepoints, t_data, '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
% ylim([0  max(model.g)*1.1]);
% yticks([]);
plot(model.tList{1}(2:end), model.f([model.f_t_list{1}]), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.tList{2}(3:end), model.f([model.f_t_list{2}(1:end-1)]), '-', 'color', [26 228 28]/255., 'LineWidth', 2.5);
ylabel('Mother');
hold off;


% WT2 Raw and fitted plot
nexttile;
hold on;
plot(model.timepoints(2, :)	, model.g(1, 16:end), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.timepoints(2, :), model.pred_g(16:end, 1), '-', 'color', [28 200 28]/255., 'LineWidth', 2.5);
ylim([0  max(model.g)*1.1]);
% yticks([]);
ylabel('WT 2');
hold off;

% Blank spot
nexttile;
plot(model.f(2:end-1), '-', 'color', [28 200 28]/255., 'LineWidth', 2.5);
%plot(model.f([model.Hpos{1}(1):1:model.Hpos{1}(2) model.Hpos{2}(1):1:model.Hpos{2}(2) ...
%	model.Hpos{4}(1):1:model.Hpos{4}(2) ]), '-', 'color', [28 200 28]/255., 'LineWidth', 2.5);


% Daughter plot
nexttile;
hold on;
plot(model.bList{1}(2:end), model.f(model.f_b_list{1}), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.bList{2}(3:end), model.f(model.f_b_list{2}(1:end-1)), '-', 'color', [26 228 28]/255., 'LineWidth', 2.5);
% ylim([0  max(model.g)*1.1]);
% yticks([]);
ylabel('Daughter');
hold off;
