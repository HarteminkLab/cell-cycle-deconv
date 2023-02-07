
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

genename = 'SIC1';

modeltype = '1.1.1';
alpha = 26;
datatype = 'JOINT';
gamma_val = 0.01;

% Find optimal gamma
model = deconvSetup(genename, modeltype, alpha);
model.gm = gamma_val;
model = deconvModel(model); 

% Plot the results
drawDeconvolved(model);

