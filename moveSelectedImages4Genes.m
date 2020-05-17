function moveSelectedImages4Genes(genefile, fromDir, toDir)

syslist = map2SystemNames(genefile, 2);

for idx=1:numel(syslist)
	gene = syslist{idx};
	deconv = strcat(gene, '.png');
	fit = strcat(gene, '-fit.png');

	movefile(strcat(fromDir, '/', deconv), strcat(toDir, '/', deconv));
	movefile(strcat(fromDir, '/', fit), strcat(toDir, '/', fit));
end

return;
