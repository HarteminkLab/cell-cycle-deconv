
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';

% Deconvolve 1 gene so we can get the dimensions of f and g easily, rather 
% than digging through the parsing code of the model file
model = Model('CLB2', 0.004);
model = deconvolve(model);

numgenes = size(stand_sys2pos, 1);
num_f = size(model.f, 1);
all_genes_f = zeros(numgenes, num_f);

num_g = size(model.g, 2);
all_genes_g = zeros(numgenes, num_g);


% Start deconvolve of all genes
[stand_sys2pos] = textread(strcat(Deconv.DECONV_DATASET, 'genes.lst'), '%s');

% Start the timer
tic;

gamma_val = 0.004;

numerror = 0;
numsuccess = 0;
for index = 1:length(stand_sys2pos)

    orf_name = stand_sys2pos{index};

    fprintf("Deconvolve %s...", orf_name);

    try
        model = Model(orf_name, gamma_val);
        all_genes_g(index, :) = model.g;

        model = deconvolve(model);
        all_genes_f(index, :) = model.f;

        numsuccess = numsuccess + 1;
    catch
        fprintf("There was an error trying to deconvolve this gene, let's skip it.");
        numerror = numerror + 1;
        continue;
    end

    % Stop the timer
    elapsedTime = toc;

    % Periodic updates
    if mod(index, 1) == 0

        fprintf("%d/%d - numerror: %d\n", index, length(stand_sys2pos), numerror);

        % Display the elapsed time
        fprintf('Elapsed time: %.4f seconds\n', elapsedTime);

        writematrix(all_genes_f, "output/all_genes_f.csv");
        writematrix(all_genes_g, "output/all_genes_g.csv");
        writecell(stand_sys2pos', "output/all_genes.csv");

        break
    end
end

writematrix(all_genes_f, "output/all_genes_f.csv");
writematrix(all_genes_g, "output/all_genes_g.csv");
writecell(stand_sys2pos', "output/all_genes.csv");
