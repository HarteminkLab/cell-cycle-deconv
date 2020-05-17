function drawReconv(one_gene, gamma);

model = deconvolve(one_gene, 'normal', 'joint', [26 27], gamma, 0, 0, 1);

c = [31 120 180]./255;

g1 = model.g(1:15);
g2 = model.g(16:30);

tp1 = model.timepoints(1,:);
tp2 = model.timepoints(2,:);
t1s = 94.387;
t1e = t1s+79.487;
t2s = 101.904;
t2e = t2s+82.014;

figure;
plot(tp1, g1, 'color', c, 'linewidth', 3);
xlim([t1s, t1e]);
ylim([0 max(g1)*1.3]);
title('WT1');

figure;
plot(tp2, g2, 'color', c, 'linewidth', 3);
xlim([t2s, t2e]);
ylim([0 max(g2)*1.3]);
title('WT2');
