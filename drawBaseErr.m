function [genes, err] = drawBaseErr(errfile, fig_flag)

genelist = '../datasets.new/commongenelist.txt';
%genelist = '../datasets.new/gene.lst';

[err, rn, sn] = textread(errfile, '%f\t%f\t%f');

if (fig_flag == 1)
	figure;
	hist(err, 200);
end

pos = find(err>0.3);
[allgenes] = textread(genelist, '%s');

genes = allgenes(pos);
err = err(pos);
