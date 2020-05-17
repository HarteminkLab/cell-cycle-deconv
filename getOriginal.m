function [g1, g2] = getOriginal(orig_gene, fig_flag);

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

DATASET_OLD = '../datasets/';
DATASET_NEW = '../datasets.new/';

DECONV_DATASET = DATASET_NEW;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');

[orfname, orfid] = map2SystemNames(orig_gene);
g1 = [];
g2 = [];
tps1 = 30:16:254;
tps2 = 38:16:262;

% ================ %
% deal with 081505 %
% ================ %
dataset = load(DATA_WT1, 'ascii');
g1 = dataset(orfid, :)';

% ================ %
% deal with 111305 %
% ================ %
dataset = load(DATA_WT2, 'ascii');
g2 = dataset(orfid, :)';

if fig_flag
	figure;
	subplot(2,1,1);
	plot(tps1, g1, '-o', 'color', 'black', 'LineWidth', 2.5);
	set(gca, 'xtick', [30,50,100,150,200,254]);
	set(gca, 'xticklabel', {'30', '50', '100', '150', '200', '254'});
	set(gca, 'ytick', []);
	set(gca, 'yticklabel', {});
%	ylim([0 1.2]);
	xlim([min(tps1), max(tps1)]);
	box off;

	subplot(2,1,2);
	plot(tps2, g2, '-o', 'color', 'black', 'LineWidth', 2.5);
	set(gca, 'xtick', [38,50,100,150,200,250, 262]);
	set(gca, 'xticklabel', {'38', '50', '100', '150', '200', '250', '262'});
	set(gca, 'ytick', []);
	set(gca, 'yticklabel', {});
%	ylim([0 1.2]);
	xlim([min(tps2), max(tps2)]);
	box off;
	suptitle(sprintf('%s', orig_gene));
end
