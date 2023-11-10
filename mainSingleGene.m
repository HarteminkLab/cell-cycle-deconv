
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))


outdir = 'output/';
plottingdir = 'output/plotting';

config = DeconvolutionConfig.yl2_replicate2_rg1_gene_expression_config();

model = Model(config, 'CLB2', 0.002);
model = deconvolve(model);

drawDeconvolvedRG1(model, plottingdir);

% --------------- Find optimal gamma through elbow method ------------

[model, flag, rn, sn, gammas, elbow_gamma] = findOptimal(model, true);

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

% ------------- Curvature --------------




