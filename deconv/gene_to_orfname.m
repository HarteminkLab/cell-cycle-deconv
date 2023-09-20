function [orfname] = gene_to_orfname(genename)


mappingfile = strcat(Deconv.DECONV_DATASET, 'gene_to_orf_name_mapping.txt');

[names, sysnames] = textread(mappingfile, '%s\t%s');
names = string(names);
sysnames = string(sysnames);
lookup = dictionary(names, sysnames);

orfname = lookup(genename);
