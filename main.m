
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv')) 
addpath(genpath('analysis'))

outdir = 'output/genes';

% Parameters
gamma_val = 0.00001;
genename = "CLB2";

fprintf("%s...", genename);
model = Model(genename, gamma_val);

fprintf("Deconvolving...");
model = deconvolve(model);

% Plot the results
fprintf("Plotting...");
fig = drawDeconvolved(model);
savename = sprintf('%s/%s.png', outdir, genename);
saveas(fig, savename); 

%close;

fprintf("Done, saved to %s\n", savename);


