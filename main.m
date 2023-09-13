
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';

genenames = ["CDC20", "SIC1", "CLN2", "PCL1", "SSK22"];
all_genes_f = 0;
all_genes_g = 0;

index = 1;
for genename = genenames

    [model] = deconvolveGene(genename);

    % lazy initialize the all genes f matrix, since we dont know the
    % size of f until we've run at least once
    if index == 1
        numgenes = size(genenames, 2);
        num_f = size(model.f, 1);
        all_genes_f = zeros(numgenes, num_f);

        num_g = size(model.g, 2);
        all_genes_g = zeros(numgenes, num_g);
    end

    all_genes_f(index, :) = model.f;
    all_genes_g(index, :) = model.g;
    index = index + 1;
end

writematrix(all_genes_f, "output/all_genes_f.csv");
writematrix(all_genes_g, "output/all_genes_g.csv");
writematrix(genenames', "output/all_genes.csv");



