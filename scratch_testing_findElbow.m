
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))


% TODO: Find optimal

genename = "CLN2";
model = Model(genename, 0.001);

% [model] = findOptimalBud(model, false);

gms = 0.001:0.005:0.2;
sns = zeros(1, size(gms, 2));
sns_nogamma = zeros(1, size(gms, 2));
rns = zeros(1, size(gms, 2));

tic;

i = 1;
for gm = gms

    gamma_val = gm;
    model = Model(genename, gamma_val);
    model = deconvolve(model);

    fprintf("For a gamma value of %3f, the rn is %.3f. ", gamma_val, model.rn);
    fprintf("The sn is %.3f, ", model.sn);
    fprintf("The sn without gamma is %.3f\n", model.sn_nogamma);

    rns(i) = model.rn;
    sns(i) = model.sn;
    % sns_nogamma(i) = model.sn_nogamma;

    i = i + 1;
end

toc;
