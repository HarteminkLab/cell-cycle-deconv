
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

genename = 'SSK22';

modeltype = '1.1.1';
alpha = 26;
datatype = 'JOINT';
gamma_val = 1.;

% Find optimal gamma
model = deconvSetup(genename, modeltype, alpha);
model.gm = gamma_val;
% model = deconvModel(model);
model = debug_f_i(model);

% Plot the results
% drawDeconvolved(model);


PADDINGSIZE = 42;
g = model.g;
mean_g = mean(g);
H = model.H;
gamma = model.gm;
Hsize = size(H, 2);

% Initial with padding to make it a power of 2
f_i = [];
last = Hsize;
f_i_front = last+[1:1:PADDINGSIZE];

last = last+PADDINGSIZE;
f_i_after = last+[1:1:PADDINGSIZE];
last = last+PADDINGSIZE;

for i = 1:length(model.i_intervals)
	idx = model.i_intervals{i}{2};
	se = model.Hpos{idx};

    fprintf("%s, interval: %d - %d\n", model.i_intervals{i}{1}, model.Hpos{idx}(1), ...
        model.Hpos{idx}(2));

	%f_i = [f_i se(1):1:se(2)];
end

f_i_pad = [f_i_front f_i f_i_after];

f_b = model.f_b;
W1 = getWaveletKernel('Symmlet', length(f_i_pad), 5);
W2 = getWaveletKernel('Symmlet', 128, 5);

% The relevant portion of f, removing the padding on the front and after
% ends, to multiply on H

f = zeros(Hsize+PADDINGSIZE*2, 1);
unpadded_f = f(PADDINGSIZE:Hsize+PADDINGSIZE-1);

fit_error = square_pos(norm(H*unpadded_f./g'-1, 2));











