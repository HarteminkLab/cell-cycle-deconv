
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/genes';

genenames = ["CLN2"];%["CLN2" "PCL1" "SIC1" "CDC20" "SSK22" "DSE1" "DSE2" "CTS1"];

% Parameters
modeltype = '1.1.1';
alphas = [26 27];
datatype = 'JOINT';
gamma_val = 0.1;

for genename = genenames
    fprintf("%s...", genename);
    fprintf("Deconvolving...");

    model = deconvSetup(genename, modeltype, alphas);
    model.gm = gamma_val;
    model = deconvModel(model); 

    fprintf("Plotting...");
    % Plot the results
    fig = drawDeconvolved(model);
    savename = sprintf('%s/%s_error_2.png', outdir, genename);
    saveas(fig, savename); 
    %close;
    %fprintf("Done, saved to %s\n", savename);
end

