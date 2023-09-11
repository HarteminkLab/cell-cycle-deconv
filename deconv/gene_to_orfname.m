function [orfname] = gene_to_orfname(genename)

[names, sysnames] = textread('datasets/original_budflow/gene_to_orf_name_mapping.txt', '%s\t%s');
names = string(names);
sysnames = string(sysnames);
lookup = dictionary(names, sysnames);

disp(genename);

orfname = lookup(genename);
