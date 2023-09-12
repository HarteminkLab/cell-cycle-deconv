
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

% Parameters
gamma_val = 0.008;
genename = "CLN2";

model = Model(genename, gamma_val);

fprintf("Deconvolving %s...", genename);
model = deconvolve(model);
fprintf("Done.\n");

[model, flag] = findOptimal(model, true);
