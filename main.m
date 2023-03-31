
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/genes';

%genenames = %;//["CLN2" "PCL1" "SIC1" "CDC20" "SSK22" "DSE1" "DSE2" "CTS1"];

% Parameters
gamma_val = 0.008;
genename = "CLN2";

fprintf("%s...", genename);
model = Model(genename, gamma_val);

fprintf("Deconvolving...");
model = deconvolve(model);

% Plot the results
fprintf("Plotting...");
fig = drawDeconvolved(model);
savename = sprintf('%s/%s.png', outdir, genename);
saveas(fig, savename); 
close;

fprintf("Done, saved to %s\n", savename);
