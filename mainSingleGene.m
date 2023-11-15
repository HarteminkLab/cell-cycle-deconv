
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))


outdir = 'output/';
plottingdir = 'output/plotting';

config = DeconvolutionConfig.xg_gene_expression_config();

gene = 'CLN2';
model = Model(config, gene, 0.00429);
model = deconvolve(model);

drawDeconvolved(model, plottingdir);

ptr = peak2troughModel(model);
fprintf("The computed PTR (80/20) for %s is: %.3f\n", gene, computedPtr);

% --------------- Find optimal gamma through elbow method ------------

plot_figure = false;

[model, flag, rn, sn, gammas, elbow_gamma] = findOptimal(model, plot_figure);

if plot_figure
    figure;

    titlename = sprintf('fit vs smooth (%s, alpha=%d, %s, gamma elbow = %0.5g)', ...
        model.orfname, model.alpha, model.datatype, elbow_gamma);
    
    title(titlename);
    
    plot(rn, sn, '--rs', 'LineWidth', 2, 'color', 'g');
    hold on;
    scatter(model.rn, model.sn, 100, 'r', 'filled');
    hold on;
    
    xlabel('fit error');
    ylabel('smooth error');
    axis square;
end

% ------------- Curvature --------------


% The published peak-to-trough ratios (PTR) for gene: YPL256C / CLN2
% 
%       Original PTR:            2.294
%       Deconvolved PTR:         4808.477
%

f_top_values = model.f(model.f_top);
f_bottom_values = model.f(model.f_bottom);






















