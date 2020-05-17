function[elbow_gamma, flag] = findElbow(model, gammas, fig_flag)

global DECONV_JOINT;
global SLIENCE;
global DEFAULT_RN_CUTOFF;

flag = 1;

if SLIENCE 
	fig_flag = 0;
end;

% residual norm (x)
rn = [];
% solution norm (y)
sn = [];

% check monotonicity
for gamma = gammas
	model.gm = gamma;
	[model] = deconvModel(model);

	if model.rn >= DEFAULT_RN_CUTOFF
		flag = 0;
		elbow_gamma = 0;
		return;
	end

	% stop if rn (residual norm) > rn_limit
	if isfield(model, 'rn_limit') && model.rn > model.rn_limit
		break;
  end
  
	if ~SLIENCE
		disp(sprintf('  ...   gamma = %0.4g, rn = %0.4g, sn = %0.4g', gamma, model.rn, model.sn));
	end
	
  rn = [rn model.rn];
  sn = [sn model.sn];
end
    
% check monotonicity
all_idx = [1];
last_idx = 1;

for idx = 2:length(sn)
	% delete if not monotonicity
	if rn(idx) >= rn(last_idx) && sn(idx) <= sn(last_idx) % OK; update
		all_idx = [all_idx idx];
		last_idx = idx;
  end
end

%figure;
%plot(rn, sn, 'o', 'color', 'r', 'linewidth', 2);

rn = rn(all_idx);
sn = sn(all_idx);
gammas = gammas(all_idx);

%hold on;
%plot(rn, sn, '--', 'color', 'b', 'linewidth', 3);

% calculate curvature

% curvature
% |x'y''-y'x''|/(x'^2+y'^2)^1.5
x_grad1 = gradient(rn);
x_grad2 = gradient(x_grad1);
y_grad1 = gradient(sn);
y_grad2 = gradient(y_grad1);
curvature = (x_grad1.*y_grad2-y_grad1.*x_grad2) ./ ((x_grad1.^2+y_grad1.^2).^(1.5));

boundary = 1;

[max_val, max_pos] = max(curvature(boundary+1:length(rn)-boundary));
max_pos = max_pos+boundary;
pos_left = max_pos-boundary;
pos_right = max_pos+boundary;
elbow_gamma = gammas(max_pos);
if numel(elbow_gamma) == 0
	flag = 0;
	elbow_gamma = 0;
	return;
end

if fig_flag
	figure;
	if strcmp(model.datatype, DECONV_JOINT)
		titlename = sprintf('fit vs smooth (%s, alpha=[%d,%d], %s, gamma elbow = %0.5g)', model.orfname, model.alpha, model.datatype, elbow_gamma);
	else
		titlename = sprintf('fit vs smooth (%s, alpha=%d, %s, gamma elbow = %0.5g)', model.orfname, model.alpha, model.datatype, elbow_gamma);
	end
	title(titlename);

	plot(rn, sn, '--rs', 'LineWidth', 2, 'color', 'g');
	hold on;
	plot(rn(pos_left:pos_right), sn(pos_left:pos_right), '--rs', 'LineWidth', 4, 'color', 'r');
	hold on;
%	plot(rn(slope_left:slope_right), sn(slope_left:slope_right), '--rs', 'LineWidth', 4, 'color', 'b');
	xlabel('fit error');
	ylabel('smooth error');
	axis square;
end
