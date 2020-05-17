%[genes, err] = drawBaseErr('baseline/allgenes.err', 0);

%output_f = './special.genes.2/special.2.clean.f';
%input_gm = './special.genes.2/special.2.clean.gm';

input_gm = './special.genes.4/special_4.gm';
output_f = './special.genes.4/special_4.f';

[orfnames, names, gammas] = textread(input_gm, '%s\t%s\t%f');

f_fid = fopen(output_f, 'w');

for i=1:length(orfnames)
	gene = orfnames{i};
	gamma = gammas(i);
	disp(sprintf('\ndeconvolving %s; gamma = %f', gene, gamma));

	model = deconvolve(gene, 'normal', 'joint', [26 27], gamma, 0, 0, 0);

	fprintf(f_fid, '%0.5g\t', model.f(1:length(model.f)-1));
	fprintf(f_fid, '%0.5g\n', model.f(end));
end

fclose(f_fid);
