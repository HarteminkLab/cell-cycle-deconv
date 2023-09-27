
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';
plottingdir = 'output/plotting';

model = Model('SSK22', 0.004);
model = deconvolve(model);
drawDeconvolvedRG1(model, plottingdir);
