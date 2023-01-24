function [colors] = getLineColors();

colors = {...
	[228 26 28], ... red (clb5)
	[255 127 0], ... orange (clb6)
	[0 222 222], ... cyan (dbf4)
	[55 126 184], ... blue (cdc7)
	[77 175 74], ... green cdc45
	[152 78 163], ... purple sld2
	[140 81 10], ... brown sld5
	[191 129 45], ... lighter psf1
	[242 184 126], ... yellow psf3
	[51 160 44]...
};

for i=1:size(colors, 2)
	colors{i} = colors{i}./255;
end

return;
