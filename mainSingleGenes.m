
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';
plottingdir = 'output/plotting';

model = Model('CLN2', 0.00554);
model = deconvolve(model);

drawDeconvolvedRG1(model, plottingdir);
