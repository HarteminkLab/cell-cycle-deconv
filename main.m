
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

% CDC20, CLB2, PCL1, SSKK22
orfnames = {'YGL116W', 'YPR119W', 'YNL289W', 'YCR073C'};

orfname = orfnames{1};

modeltype = '1.1.1';
alpha = 26; 
datatype = 'WT1';
gamma = 0.012;

% Find optimal gamma
model = deconvSetup(orfname, modeltype, alpha);

% model = findOptimal(model, 0);

model.gm = gamma;
model = deconvModel(model);

model = getRDCidx(model);

fig1 = figure;
pbaspect([16 9 1]);
hold on;
box on;
WT1_TP = 30:16:254;
plot(WT1_TP, model.g, '-', 'color', [228 26 28]/255., 'LineWidth', 2.5);
title("Raw");
plot(WT1_TP, model.H*model.f, '-', 'color', [28 200 28]/255., 'LineWidth', 2.5);
title("Fit");
saveas(gcf, strcat('output/', orfname, '_fit'), 'png');
hold off;
box off;

drawDeconvolved(model);

