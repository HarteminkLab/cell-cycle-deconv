%genes = {'CDC6', 'CHS1', 'TCB2', 'HTB2', 'HSP12', 'SIC1', 'ASH1', 'AMN1', 'CLB5', 'CLN2', 'CLB4', 'CLB1', 'CLB2', 'ACT1'};
%models = {'DIFF_G1', 'RDG1', 'RCG1', 'NORMAL', 'ELDG1', 'SRDG1', 'SRCG1', 'SCG1R', 'SDG1R'};

genes = {'HSP12', 'CLB5', 'HTB2', 'CLB4', 'CLB2', 'SIC1', 'CDC6', 'CHS1', 'TCB2'};
models = {'DG1', 'CG1', 'SCG1DG1'};

gm_file = 'allgenes/allgenes.gm';

output = 'fit_stat_SCG1R.tab';
fid = fopen(output, 'w');

for a = 1:length(models)
	fprintf(fid, '\t%s', models{a});
end
fprintf(fid, '\n');

gm = 0;

for g = 1:length(genes)
	cur_gene = genes{g};
	disp(sprintf('gene = %s', cur_gene));
	fprintf(fid, '%s', cur_gene);

	for m = 1:length(models)
		cur_model = models{m};
		disp(sprintf('\tmodel = %s, gm = %f', cur_model, gm));
		model = deconvolve(cur_gene, cur_model, 'joint', [26 27], gm, 0, 0, 0);

		fiterror = square_pos(norm(log(model.H*model.f)-log(model.g),2));
		fprintf(fid, '\t%0.3g', fiterror);
	end
	fprintf(fid, '\n');
end

fclose(fid);
