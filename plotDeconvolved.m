function [model] = plotDeconvolved(model, plottingdir)
    
    % Make the plotting directory
    try
        mkdir(plottingdir);
    catch
        % Skip, make the directory silently
    end
    
    fprintf("The rn of the model is: %.4f\n", model.rn);
    fprintf("The sn of the model is: %.4f\n", model.sn);
    
    % Plot the results
    fprintf("Plotting...");
    fig = drawDeconvolved(model);
    savename = sprintf('%s/%s.png', plottingdir, model.genename);
    saveas(fig, savename); 

    % close;

    fprintf("Done, saved to %s\n", savename);

end