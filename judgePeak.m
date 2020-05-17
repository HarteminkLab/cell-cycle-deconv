function [flag, peak_idx] = judgePeak(model, bin);

% flag:
% 1. C
% 2. D

if nargin < 2;
	bin = 100;
end

t_x = model.t_x;
t_y = model.f(model.t_y_idx);
b_x = model.b_x;
b_y = model.f(model.b_y_idx);

[t_yr, t_xr] = rescale(t_x, t_y, (t_x(end)-t_x(1))/(bin-1));
[b_yr, b_xr] = rescale(b_x, b_y, (b_x(end)-b_x(1))/(bin-1));

[vc, peak_c] = max(t_yr);
[vd, peak_d] = max(b_yr);

if vc >= vd;
	flag = 1;
	peak_idx = peak_c;
else
	flag = 2;
	peak_idx = peak_d;
end;

return;
