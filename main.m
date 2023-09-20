
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';

[stand_sys2pos] = textread(strcat(Deconv.DECONV_DATASET, 'genes.lst'), '%s');

all_genes_f = 0;
all_genes_g = 0;

% Start the timer
tic;

numerror = 0;
numsuccess = 0;
for index = 1:length(stand_sys2pos)

    orf_name = stand_sys2pos{index};

    try
        [model] = deconvolveGene(orf_name);
        numsuccess = numsuccess + 1;
    catch
        % fprintf("There was an error trying to deconvolve this gene, let's skip it.");
        numerror = numerror + 1;
        continue;
    end

    % lazy initialize the all genes f matrix, since we dont know the
    % size of f until we've run at least once
    if numsuccess == 1
        numgenes = size(stand_sys2pos, 2);
        num_f = size(model.f, 1);
        all_genes_f = zeros(numgenes, num_f);

        num_g = size(model.g, 2);
        all_genes_g = zeros(numgenes, num_g);
    end

    all_genes_f(index, :) = model.f;
    all_genes_g(index, :) = model.g;

    % Stop the timer
    elapsedTime = toc;

    % Periodic updates
    if mod(numsuccess, 100) == 0

        fprintf("%d/%d - numerror: %d\n", index, length(stand_sys2pos), numerror);

        % Display the elapsed time
        fprintf('Elapsed time: %.4f seconds\n', elapsedTime);

        writematrix(all_genes_f, "output/all_genes_f.csv");
        writematrix(all_genes_g, "output/all_genes_g.csv");
        writecell(stand_sys2pos', "output/all_genes.csv");
    end
end

writematrix(all_genes_f, "output/all_genes_f.csv");
writematrix(all_genes_g, "output/all_genes_g.csv");
writecell(stand_sys2pos', "output/all_genes.csv");
