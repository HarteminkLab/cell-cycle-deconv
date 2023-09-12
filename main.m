
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';

genenames = ["CLN2"];

% Parameters
gamma_val = 0.008;
genename = "CLN2";

model = Model(genename, gamma_val);

fprintf("Deconvolving %s...", genename);
model = deconvolve(model);
fprintf("Done.\n");

% Make the plotting directory
plottingdir = 'output/plotting';
try
    mkdir(plottingdir);
catch
    % Skip, make the directory silently
end


% Plot the results
fprintf("Plotting...");
fig = drawDeconvolved(model);
savename = sprintf('%s/%s.png', plottingdir, genename);
saveas(fig, savename); 

close;

fprintf("Done, saved to %s\n", savename);
