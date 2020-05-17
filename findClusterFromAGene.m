function [list, list_G1, list_G2]= findClusterFromAGene(gene, n);

global DECONV_DATASET;

DATASET_NEW = '../datasets.new/';

DECONV_DATASET = DATASET_NEW;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');
ALL_GENE = strcat(DECONV_DATASET, 'commongenelist.txt');

allgenes = textread(ALL_GENE, '%s');

[sysnames, sysid, genes] = map2SystemNames(gene);
tps1 = 30:16:254;
tps2 = 38:16:262;

dataset1 = load(DATA_WT1, 'ascii');
dataset2 = load(DATA_WT2, 'ascii');
g_wt1 = dataset1(sysid, :);
g_wt2 = dataset2(sysid, :);

score = [];

num_all = numel(allgenes);
for i=1:num_all
	cur_g1 = dataset1(i, :);
	cur_g2 = dataset2(i, :);
	s1 = corr2(cur_g1, g_wt1);
	s2 = corr2(cur_g2, g_wt2);
	score = [score (s1+s2)/2];
end

[score, order] = sort(score, 'descend');
allgenes = allgenes(order);

dataset1 = dataset1(order,:);
dataset2 = dataset2(order,:);

list = allgenes(2:n+1);
list_G1 = dataset1(2:n+1, :);
list_G2 = dataset2(2:n+1, :);

figure;

subplot(2,1,1);
plot(tps1, g_wt1, '-o', 'linewidth', 1, 'color', 'r');
hold on;
for i=1:n
	plot(tps1, list_G1(i,:), '-o', 'linewidth', 1, 'color', [129 129 129]./255);
	hold on;
end
subplot(2,1,2);
plot(tps2, g_wt2, '-o', 'linewidth', 1, 'color', 'r');
hold on;
for i=1:n
	plot(tps2, list_G2(i,:), '-o', 'linewidth', 1, 'color', [129 129 129]./255);
	hold on;
end

return;
