
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';
plottingdir = 'output/plotting';

[model] = deconvolveGene('CLB2');
[model] = plotDeconvolved(model, plottingdir);
