function [] = main()

    outdir = 'output/2023-10-17_yl_rep2_entropy_nuc_genebody_rg1_sigma0_11/';
    
    addpath(genpath('lib/YAMLMatlab'));
    addpath(genpath('deconv'))
    addpath(genpath('analysis'))

    % Deconvolve 1 gene so we can get the dimensions of f and g easily, rather 
    % than digging through the parsing code of the model file
    model = Model('CLB2', 0.004);
    model = deconvolve(model);

    writematrix(model.H, strcat(outdir, '/H.csv'));

    % Start deconvolve of all genes
    [stand_sys2pos] = textread(Deconv.GENESET_PATH, '%s');
    
    numgenes = size(stand_sys2pos, 1);
    num_f = size(model.f, 1);
    all_genes_f = zeros(numgenes, num_f);
    
    num_g = size(model.g, 2);
    all_genes_g = zeros(numgenes, num_g);
    
    % Start the timer
    tic;
    
    gamma_val = 0.004;
    
    numerror = 0;
    numsuccess = 0;
    for index = 1:length(stand_sys2pos)
    
        orf_name = stand_sys2pos{index};
    
        elapsedTime = toc;
        fprintf("Deconvolving %s (%d/%d) - %.3f min\n", orf_name, index, size(stand_sys2pos, 1), ...
            elapsedTime/60.);
    
        try
            model = Model(orf_name, gamma_val);
            all_genes_g(index, :) = model.g;
    
            model = deconvolve(model);
            all_genes_f(index, :) = model.f;
    
            numsuccess = numsuccess + 1;
        catch
            fprintf("   There was an error trying to deconvolve this gene, let's skip it.\n");
            numerror = numerror + 1;
            continue;
        end
    
        % Stop the timer
        elapsedTime = toc;
    
        % Periodically save the output filescomcomc
        if mod(index, 10) == 0
            writedata(outdir, all_genes_f, all_genes_g, stand_sys2pos);
        end
    end
    
    writedata(outdir, all_genes_f, all_genes_g, stand_sys2pos);
    
    fprintf("Done. Completed in %.3f min\n", elapsedTime/60.);
end

function[] = writedata(outdir, all_genes_f, all_genes_g, stand_sys2pos)

    writematrix(all_genes_f, strcat(outdir, "/all_genes_f.csv"));
    writematrix(all_genes_g, strcat(outdir, "/all_genes_g.csv"));
    writecell(stand_sys2pos', strcat(outdir, "/all_genes.csv"));

end
