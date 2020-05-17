function [W] = drawW(WAVETYPE, len, WAVEPAR)


% ============ %
% == draw W == %
% ============ %
W = getWaveletKernel(WAVETYPE, len, WAVEPAR);

%fig1 = figure;
%set(fig1, 'OuterPosition', [400 50 800 1100]);

%maxW = max(max(W));

frames = 4;
for i=1:len
	if mod(i, frames) == 1
		fig1 = figure;
		set(fig1, 'OuterPosition', [400 50 800 1100]);
	end
	y = W(i, :);

	set(gca, 'ytick', 0);
	set(gca, 'xtick', []);

	subplot(frames,1, mod(i-1, frames)+1);
	plot(y);

	xlim([0 numel(y)]);
%	ylim([0 maxW*1.2]);
	set(gca, 'yticklabel', i);
end

return;
