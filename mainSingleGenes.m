
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';

[model] = deconvolveGene('CLB2');
