W = getWaveletKernel('Symmlet', 128, 5);

all = [];
for i=1:128;
	row = W(i,:);
%	diff = abs(row(1)-row(end));
%	all = [all diff];
	figure;
	plot(row);
	k = waitforbuttonpress ;
	close;
end
%figure;
%plot(all);
