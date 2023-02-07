function[model] = drawDeconvolved(model)

f = figure();
f.Position = [100 100 1200 600];
tiled_layout = tiledlayout(3, 3);
tiled_layout.Padding = 'compact';
tiled_layout.TileSpacing = 'compact';
title(tiled_layout, model.genename);

plot_g = model.g;
plot_pred_g = model.pred_g;
plot_f = model.f;

ylim_raw = max(plot_g)*1.25;
ylim_deconv = max(plot_f)*1.25;

% WT1 Raw and fitted plot
nexttile;
hold on;
plot(model.timepoints(1, :), plot_g(1, 1:15), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.timepoints(1, :), plot_pred_g(1:15, 1), '-', 'color', [28 200 28]/255., 'LineWidth', 2.5);
ylim([0  ylim_raw]);
yticks([]);
ylabel('WT 1');
hold off;

% Initial plot
nexttile;
hold on;
plot(model.iList{1}(2:end), plot_f(model.f_i_list{1}(1:end)), '-', 'color', [28 26 228]/255., 'LineWidth', 2.5);
plot(model.iList{2}(2:end), plot_f(model.f_i_list{2}(1:end)), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.iList{3}(2:end), plot_f(model.f_i_list{3}(1:end)), '-', 'color', [26 228 28]/255., 'LineWidth', 2.5);
ylim([0  ylim_deconv]);
yticks([]);
ylabel('Initial: R, CG1, postG1; f_i');
hold off;

% Mother plot
nexttile;
hold on;
ylim([0  ylim_deconv]);
yticks([]);
plot(model.tList{1}(2:end), plot_f([model.f_t_list{1}]), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.tList{2}(2:end), plot_f([model.f_t_list{2}]), '-', 'color', [26 228 28]/255., 'LineWidth', 2.5);
ylabel('Top/Mother: CG1, postG1; f_t');
xlim([min(model.tList{1}) max(model.tList{2})]);
hold off;


% WT2 Raw and fitted plot
nexttile;
hold on;
plot(model.timepoints(2, :)	, plot_g(1, 16:end), '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
plot(model.timepoints(2, :), plot_pred_g(16:end, 1), '-', 'color', [28 200 28]/255., 'LineWidth', 2.5);
ylim([0  ylim_raw]);
yticks([]);
ylabel('WT 2');
hold off;

% f vector
nexttile;
hold on;
ylim([0  ylim_deconv]);
yticks([]);
ylabel('f: R, CG1, DG1, postG1');
colors = {'blue', 'red', 'magenta', 'green'};
abs_indices = 1:1:size(model.H, 2);
for i = 1:length(model.Hpos)
    indices = model.Hpos{i}(1):1:model.Hpos{i}(2);
    plot(abs_indices(indices), model.f(indices), '-', 'color', colors{i}, 'LineWidth', 2.5);
end
xlim([1 size(model.H, 2)]);
hold off;

% Daughter plot
nexttile;
hold on;
plot(model.bList{1}(2:end), plot_f(model.f_b_list{1}), '-', 'color', 'magenta', 'LineWidth', 2.5);
plot(model.bList{2}(3:end), plot_f(model.f_b_list{2}(1:end-1)), '-', 'color', [26 228 28]/255., 'LineWidth', 2.5);
ylim([0  ylim_deconv]);
yticks([]);
xlim([min(model.bList{1}) max(model.bList{2})]);
ylabel('Bottom/Daughter: DG1, postG1; f_b');
hold off;

nexttile;
axis off;

% g heatmap
nexttile;
hm = heatmap(model.H);
hm.GridVisible = 'off';
colorbar();
ylabel('g: R, CG1, DG1, postG1');
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));

