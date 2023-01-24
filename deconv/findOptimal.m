function [model, flag] = findOptimal(model, fig_flag)

gammas = 0.001:0.001:0.02;
rns = zeros(size(gammas));
sns = zeros(size(gammas));

for i = 1:size(gammas, 2)
    cur_gm = gammas(i);
    model.gm = cur_gm;
    [model] = deconvModel(model);
    rns(i) = model.rn;
    sns(i) = model.sn;
    fprintf("%.3f - rn=%.3f, sn=%.3f\n", cur_gm, model.rn, model.sn);
end

figure;
pbaspect([16 9 1]);
hold on;
box on;
plot(gammas, rns);
title("Residual Norms");
saveas(gcf, strcat('output/', model.orfname, '_rns'), 'png');
box off;
hold off;

figure;
hold on;
box on;
pbaspect([16 9 1]);
plot(gammas, sns);
title("Solution Norms");
saveas(gcf, strcat('output/', model.orfname, '_sns'), 'png');
box off;
hold off;

model.gm = 0.007;
