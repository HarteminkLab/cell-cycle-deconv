[names, corr, base] = textread('allgenes.baseline.err', '%s\t%f\t%f');
output_good = 'goodGenesOnBaseline.lst';
output_notgood = 'notGoodGenesOnBaseline.lst';

cut = 10;

good_fid = fopen(output_good, 'w');
notgood_fid = fopen(output_notgood, 'w');

good = names(find(base<=cut));
notgood = names(find(base>cut));

for i=1:numel(good)
	fprintf(good_fid, '%s\n', good{i});
end

for i=1:numel(notgood)
	fprintf(notgood_fid, '%s\n', notgood{i});
end

fclose(good_fid);
fclose(notgood_fid);
