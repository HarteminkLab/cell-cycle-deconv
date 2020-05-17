function [W] = addWeightToMatrix(W, f)

for i=1:size(W,2)
	W(:,i) = W(:,i)*f(i);
end
