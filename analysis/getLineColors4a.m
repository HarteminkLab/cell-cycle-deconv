function [colors] = getLineColors();

colors = {...
	[228 26 28], ... red mcm2
	[255 127 0], ... orange mcm3
	[255 196 81], ... yellow mcm5
	[55 126 184], ... blue mcm4
	[77 175 74], ... green mcm6
	[152 78 163], ... purple mcm7
	[166 86 40], ... brown cdc6
	[127 127 127], ... dark black
	[247 129 191], ... pink
	[153 153 153], ... grey
	[0 109 44], ...
	[31 120 180], ...
	[178 223 138], ...
	[253 191 111], ...
	[202 178 214], ...
	[51 160 44]...
};

for i=1:size(colors, 2)
	colors{i} = colors{i}./255;
end

return;
