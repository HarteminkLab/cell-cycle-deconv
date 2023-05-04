function[elbow_gamma, flag, rn, gammas] = findElbow(model, gammas)

global DECONV_JOINT;
global DEFAULT_RN_CUTOFF;

flag = 1;

% residual norm (x)
% solution norm (y)
rn = [];
sn = [];

% check monotonicity
for gamma = gammas

	model.gm = gamma;
	[model] = deconvolve(model);

	if model.rn >= DEFAULT_RN_CUTOFF
		flag = 0;
		elbow_gamma = 0;
		return;
	end

	% stop if rn (residual norm) > rn_limit
	if isfield(model, 'rn_limit') && model.rn > model.rn_limit
		break;
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

rn = rn(all_idx);
sn = sn(all_idx);
gammas = gammas(all_idx);

% curvature
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
