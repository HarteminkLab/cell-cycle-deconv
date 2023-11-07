
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))


% TODO: Find optimal

genename = "CLN2";
model = Model(genename, 0.001);

tic;

[gammas, rn, sn, final_gm] = findOptimal_small(model, true);

toc;
