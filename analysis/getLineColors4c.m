function [colors] = getLineColors();

colors = {...
	[228 26 28], ... red hta1
	[255 127 0], ... orange hta2
	[255 196 81], ... yellow htb1
	[77 175 74], ... green htb2
	[0 222 222], ... cyan hht1
	[55 126 184], ... blue hht2
	[152 78 163], ... purple hhf1/2
	[166 86 40], ... grey
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
