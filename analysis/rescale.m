function [newy, newx] = rescale(x, y, interval)
% x: old x
% y: old y
% interval in new x, default is 1

if nargin < 2
	display('... Expect at least 2 inputs: x, y, interval (optional)');
	return;
end

if nargin == 2
	interval = 1;
end

len = ceil((x(end)-x(1))/interval);
newx = [];
newy = [];
cur_x = x(1);
for idx = 1:len
	% find the left and the right neighbours
	left_pos = find(x<=cur_x, 1, 'last');
	right_pos = find(x>=cur_x, 1, 'first');

	x_left = x(left_pos);
	y_left = y(left_pos);
	x_right = x(right_pos);
	y_right = y(right_pos);

	newx = [newx cur_x];
	if x_left == x_right
		newy(idx) = y_left;
	else
		newy(idx) = y_left+(cur_x-x_left)/(x_right-x_left)*(y_right-y_left);
	end
	cur_x = cur_x + interval;
end

newx = [newx newx(end)+interval];
newy = [newy newy(end)];

return;
