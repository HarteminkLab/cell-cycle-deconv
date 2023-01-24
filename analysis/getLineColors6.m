function [colors] = getLineColors();

colors = {...
	[247 129 191], ... pink ASH1
	[228 26 28], ... red egt2 
	[255 127 0], ... orange amn1
	[255 196 81], ... yellow dse3
	[55 126 184], ... blue dse4
	[77 175 74], ... green pry3
	[152 78 163], ... purple scw11
	[166 86 40], ... brown dse1
	[127 127 127], ... dark dse2
	[0 90 50]...
	[178 223 138], ... CTS1
	[153 153 153], ... grey
	[0 109 44], ... 
	[31 120 180], ... CTS1
	[253 191 111], ...
	[202 178 214], ...
};

for i=1:size(colors, 2)
	colors{i} = colors{i}./255;
end

return;
