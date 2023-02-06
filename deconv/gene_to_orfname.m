function [orfname] = gene_to_orfname(genename)

[names, sysnames] = textread('datasets/map2sys2.txt', '%s\t%s');
names = string(names);
sysnames = string(sysnames);
lookup = dictionary(names, sysnames);
orfname = lookup(genename);
