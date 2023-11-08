
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

outdir = 'output/';
plottingdir = 'output/plotting';

model = Model('CLB2', 0.002);
model = deconvolve(model);

drawDeconvolvedRG1(model, plottingdir);

% --------------- Find optimal gamma through elbow method ------------

[gammas, rn, sn, elbow_gamma] = findOptimal(model, true);

figure;

titlename = sprintf('fit vs smooth (%s, alpha=%d, %s, gamma elbow = %0.5g)', model.orfname, model.alpha, model.datatype, elbow_gamma);

title(titlename);

plot(rn, sn, '--rs', 'LineWidth', 2, 'color', 'g');
hold on;
scatter(model.rn, model.sn, 100, 'r', 'filled');
hold on;

xlabel('fit error');
ylabel('smooth error');
axis square;

% ------------- Curvature --------------

x_grad1 = gradient(rn);
x_grad2 = gradient(x_grad1);
y_grad1 = gradient(sn);
y_grad2 = gradient(y_grad1);
curvature = (x_grad1.*y_grad2-y_grad1.*x_grad2) ./ ((x_grad1.^2+y_grad1.^2).^(1.5));

boundary = 1;

fprintf("The x_grad1 is: %f\n", x_grad1);
fprintf("The x_grad2 is: %f\n", x_grad2);
fprintf("The curvature is: %f\n", curvature);

[max_val, max_pos] = max(curvature(boundary+1:length(rn)-boundary));
max_pos = max_pos+boundary;
pos_left = max_pos-boundary;
pos_right = max_pos+boundary;
elbow_gamma = gammas(max_pos);

figure;
plot(x_grad1);




