function [orfname] = gene_to_orfname(genename)


mappingfile = Deconv.GENEMAPPING_PATH;

[names, sysnames] = textread(mappingfile, '%s\t%s');
names = string(names);
sysnames = string(sysnames);
lookup = dictionary(names, sysnames);

orfname = lookup(genename);
