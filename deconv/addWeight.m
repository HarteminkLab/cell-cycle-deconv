function [W] = addWeight(W, f)

global NORMALIZING;

if ~NORMALIZING
	return;
end

f_ele = myunique(f);

display(strcat('addWeight: [', num2str(f_ele), ']'));

for i=1:size(W,2)
	W(:,i) = W(:,i)*f(i);
end
