function val = myMass(y, x)

val = 0;
cury = y(1);
for i=2:numel(x)
	delta_x = x(i)-x(i-1);
	delta_y = y(i)-y(i-1);

	val = val+delta_x*cury+delta_x*delta_y/2;
	cury = y(i);
end

return;
