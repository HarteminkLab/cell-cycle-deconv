%genes = {'CDC6', 'CHS1', 'TCB2', 'HTB2', 'HSP12', 'SIC1', 'ASH1', 'AMN1', 'CLB5', 'CLN2', 'CLB4', 'CLB1', 'CLB2', 'ACT1'};
%models = {'DIFF_G1', 'RDG1', 'RCG1', 'NORMAL', 'ELDG1', 'SRDG1', 'SRCG1', 'SCG1R', 'SDG1R'};

genes = {'HSP12', 'CLB5', 'HTB2', 'CLB4', 'CLB2', 'SIC1', 'CDC6', 'CHS1', 'TCB2'};
models = {'DG1', 'RG1', 'SCG1DG1'};

gm_file = 'allgenes/allgenes.gm';

[allgenes, allnames, allgm] = textread(gm_file, '%s\t%s\t%f');

for g = 1:length(genes)
	cur_gene = genes{g};
	[orfname, orfid] = map2SystemNames(cur_gene);
	cur_gm = allgm(orfid);

	disp(sprintf('gene = %s', cur_gene));

	for m = 1:length(models)
		cur_model = models{m};
		disp(sprintf('\tmodel = %s, gm = %f', cur_model, cur_gm));
		model = deconvolve(cur_gene, cur_model, 'joint', [26 27], cur_gm, 0, 0, 1);
	end
end
