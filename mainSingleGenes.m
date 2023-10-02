
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';
plottingdir = 'output/plotting';

model = Model('CLB2', 0.004);
model = deconvolve(model);
drawDeconvolvedRG1(model, plottingdir);

writematrix(model.H, strcat(outdir, '/H.csv'));