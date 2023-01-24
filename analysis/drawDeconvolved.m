function[model] = drawDeconvolved(model)

b_timepoints = [model.bList{1} model.bList{2}];
b_timepoints = b_timepoints(1:end, 3:end)';
t_timepoints = [model.tList{1} model.tList{2}];
t_timepoints = t_timepoints(1:end, 3:end)';

b_data = model.f(model.b_y_idx);
t_data = model.f(model.t_y_idx);

it_data = [model.f(model.i_y_idx)' model.f(model.t_y_idx)'];

figure;
plot(b_timepoints, b_data, '-', 'color', [140 202 222]/255., 'LineWidth', 2.5);
hold on;
pbaspect([16 9 1]);
title("Daughter");
ylim([0  max(b_data)*1.1]);
saveas(gcf, strcat('output/', model.orfname, '_daughter'), 'png');


figure;
plot(t_timepoints, t_data, '-', 'color', [140 202 222]/255., 'LineWidth', 2.5);
hold on;
pbaspect([16 9 1]);
title("Mother");
ylim([0  max(t_data)*1.1]);
saveas(gcf, strcat('output/', model.orfname, '_daughter'), 'png');
