
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/genes';

%genenames = %;//["CLN2" "PCL1" "SIC1" "CDC20" "SSK22" "DSE1" "DSE2" "CTS1"];

% Parameters
gamma_val = 0.008;
genename = "CLB2";

mappingfile = 'datasets/original_budflow/map2sys2.txt';

modelpath1 = 'models/original_budflow/wt1_budflow/1.1.1.26.label';
modelpath2 = 'models/original_budflow/wt2_budflow/1.1.1.27.label';

fprintf("%s...", genename);
model = Model(genename, gamma_val, modelpath1, modelpath2, mappingfile);

fprintf("Deconvolving...");
model = deconvolve(model);

% Plot the results
fprintf("Plotting...");
fig = drawDeconvolved(model);
savename = sprintf('%s/%s.png', outdir, genename);
saveas(fig, savename); 

close;

fprintf("Done, saved to %s\n", savename);


