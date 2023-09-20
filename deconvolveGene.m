function [model] = deconvolveGene(genename)

    % Parameters
    gamma_val = 0.004;
    
    model = Model(genename, gamma_val);
    
    % Try logging the gene expression and let's see why the rn remains so high
    % model.g = log(model.g);
    
    fprintf("Deconvolving %s...", genename);
    model = deconvolve(model);
    fprintf("Done.\n");

end
