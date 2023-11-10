function[elbow_gamma] = findElbowBud(model, gammas, fig_flag)

global DECONV_JOINT;
% shut off fig_flag
fig_flag = 1;

% residual norm (x)
rn = [];
% solution norm (y)
sn = [];

% check monotonicity
for gamma = gammas
	model.gm = gamma;
	[model] = deconvolve(model);

	% stop if rn (residual norm) > rn_limit
	if isfield(model, 'rn_limit') && model.rn > model.rn_limit
		break;
    end
    
    rn = [rn model.rn];
    sn = [sn model.sn];
end
    
% check monotonicity
% try to find longest region with monotonicity

prev_rn = [];
prev_sn = [];
prev_gm = [];

cur_rn = [rn(1)];
cur_sn = [sn(1)];
cur_gm = [gammas(1)];

for idx = 2:length(sn)
	% truncate if not monotonicity
	if rn(idx) >= rn(idx-1) && sn(idx) <= sn(idx-1)
		cur_rn = [cur_rn rn(idx)];
		cur_sn = [cur_sn sn(idx)];
		cur_gm = [cur_gm gammas(idx)];
	else
		if length(cur_rn) > length(prev_rn)
				prev_rn = cur_rn;
				prev_sn = cur_sn;
				prev_gm = cur_gm;
		end 
	
    cur_rn = [rn(idx)];
		cur_sn = [sn(idx)];
		cur_gm = [gammas(idx)];
  end
end

if length(cur_rn) > length(prev_rn)
	prev_rn = cur_rn;
	prev_sn = cur_sn;
	prev_gm = cur_gm;
end

rn = prev_rn;
sn = prev_sn;
local_gm = prev_gm;

% calculate curvature
% in log-space
rn_log = log10(rn);
sn_log = log10(sn);

% curvature
% |x'y''-y'x''|/(x'^2+y'^2)^1.5
x_grad1 = gradient(rn_log);
x_grad2 = gradient(x_grad1);
y_grad1 = gradient(sn_log);
y_grad2 = gradient(y_grad1);
curvature = (x_grad1.*y_grad2-y_grad1.*x_grad2) ./ ((x_grad1.^2+y_grad1.^2).^(1.5));

win_size = 0;

[max_val, max_pos] = max(curvature(win_size+1:length(rn)-win_size));
max_pos = max_pos+win_size;
pos_left = max_pos-win_size;
pos_right = max_pos+win_size;
elbow_gamma = local_gm(max_pos);

if fig_flag
	figure;
	if strcmp(model.datatype, DECONV_JOINT)
		titlename = sprintf('fit vs smooth (%s, alpha=[%d,%d], %s, gamma elbow = %0.5g)', model.orfname, model.alpha, model.datatype, elbow_gamma);
	else
		titlename = sprintf('fit vs smooth (%s, alpha=%d, %s, gamma elbow = %0.5g)', model.orfname, model.alpha, model.datatype, elbow_gamma);
	end
	title(titlename);

	plot(exp(rn_log), exp(sn_log), '--rs', 'LineWidth', 2, 'color', 'g');
	hold on;
	plot(exp(rn_log(pos_left:pos_right)), exp(sn_log(pos_left:pos_right)), '--rs', 'LineWidth', 4, 'color', 'r');
	hold on;
%	plot(rn_log(slope_left:slope_right), sn_log(slope_left:slope_right), '--rs', 'LineWidth', 4, 'color', 'b');
	xlabel('fit error');
	ylabel('smooth error');
	axis square;
end
