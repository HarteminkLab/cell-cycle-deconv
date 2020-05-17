function getDeconvolvedProfileFromList(geneshortlist, f_in, gm_in, f_out, gm_out);

Hall = load(f_in);
[a, b, c] = textread(gm_in, '%s\t%s\t%f');

orig_genes = textread(geneshortlist, '%s');
[genes, orfids] = map2SystemNames(orig_genes);


fid = fopen(f_out, 'w');
gid = fopen(gm_out,'w');

for idx = 1:length(genes)
	orfid = orfids{idx};
	curf = Hall(orfid,:);
	fprintf(fid, '%0.5g\t', curf(1:end-1));
	fprintf(fid, '%0.5g\n', curf(end));
	fprintf(gid, '%s\t%s\t%0.5g\n', a{orfid}, b{orfid}, c(orfid));
end

fclose(fid);
fclose(gid);

return;
