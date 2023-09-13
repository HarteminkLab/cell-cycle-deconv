function [model] = deconvolveGene(genename)

    % Parameters
    gamma_val = 0.004;
    
    model = Model(genename, gamma_val);
    
    % Try logging the gene expression and let's see why the rn remains so high
    % model.g = log(model.g);
    
    fprintf("Deconvolving %s...", genename);
    model = deconvolve(model);
    fprintf("Done.\n");
    
    % Make the plotting directory
    plottingdir = 'output/plotting';
    try
        mkdir(plottingdir);
    catch
        % Skip, make the directory silently
    end
    
    fprintf("The rn of the model is: %.4f\n", model.rn);
    fprintf("The sn of the model is: %.4f\n", model.sn);
    
    % Plot the results
    % fprintf("Plotting...");
    % fig = drawDeconvolved(model);
    % savename = sprintf('%s/%s.png', plottingdir, genename);
    % saveas(fig, savename); 
    % 
    % close;
    % 
    % fprintf("Done, saved to %s\n", savename);

end
