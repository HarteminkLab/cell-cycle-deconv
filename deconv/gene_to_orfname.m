function [orfname] = gene_to_orfname(config, genename)


mappingfile = config.GENEMAPPING_PATH;

[names, sysnames] = textread(mappingfile, '%s\t%s');
names = string(names);
sysnames = string(sysnames);
lookup = dictionary(names, sysnames);

orfname = lookup(genename);
