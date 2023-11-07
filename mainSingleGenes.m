
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';
plottingdir = 'output/plotting';

model = Model('CLN2', 0.04);
model = deconvModel(model);

% drawDeconvolvedRG1(model, plottingdir);
%writematrix(model.H, strcat(outdir, '/H.csv'));
