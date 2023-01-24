function [model, runs, flag] = binarysearch(model, gamma_min, gamma_max, rn_goal)
global SLIENCE;
global DEFAULT_RN_CUTOFF;

RN_SMALL = 2e-4;
LR_SMALL = 5e-4;
flag = 1;

left = gamma_min;
right = gamma_max;
runs = 0;
% find the fit_left point
while right-left > LR_SMALL
	cur_gamma = (left+right)/2;
	runs = runs+1;
	
	model.gm = cur_gamma;
	
	[model] = deconvModel(model);

	if model.rn >= DEFAULT_RN_CUTOFF
		flag = 0;
		return;
	end

	if abs(model.rn-rn_goal) <= RN_SMALL
		break;
	elseif model.rn > rn_goal
		right = cur_gamma;
	else
		left = cur_gamma;
	end

	rn_rate = (model.rn/model.base_rn-1)*100;
	if ~SLIENCE
		disp(sprintf('  ...   gm = %0.4f, rn = %0.4f, rate = %0.1f', model.gm, model.rn, rn_rate));
	end

end
return;