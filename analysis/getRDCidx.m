function [model] = getRDCidx(model);

global DECONV_DELTAPOS;

% get intervals and labels from model structure
t_x = [];
b_x = [];
i_x = [];
r_x = [];

dg1_x = [];	% dg1
cg1_x = [];	% cg1
d_x = [];		% d (dg1)
g1_x = [];	% g1 (dg1)

postg1_x = [];
s_x = [];
g2m_x = [];

t_y_idx = [];
b_y_idx = [];
i_y_idx = [];
r_y_idx = [];

dg1_y_idx = [];
cg1_y_idx = [];
d_y_idx = []; 
g1_y_idx = [];

postg1_y_idx = [];
s_y_idx = [];
g2m_y_idx = [];

s_phase = 0.2;

% b_x = [0 delta+lambda]
offset_b_min = inf;
for i = 1:length(model.bList)
	offset_b_min = min(offset_b_min, min(model.bList{i}'));
end
for i = 1:length(model.bList)
	list = model.bList{i}';
	b_x = [b_x; list(1:length(list)-1)-offset_b_min];
end

% t_x = [0 lambda]
offset_t_min = inf;
for i = 1:length(model.tList)
	offset_t_min = min(offset_t_min, min(model.tList{i}'));
end
for i = 1:length(model.tList)
	list = model.tList{i}';
	t_x = [t_x; list(1:length(list)-1)-offset_t_min];
end

% i_x = [0 mu0+lambda-alpha]
offset_i_min = inf;
for i = 1:length(model.iList)
	offset_i_min = min(offset_i_min, min(model.iList{i}'));
end
for i = 1:length(model.iList)
	list = model.iList{i}';
	i_x = [i_x; list(1:length(list)-1)-offset_i_min];
end

% r_x = [0 mu0-alpha]
offset_r_min = inf;
for i = 1:length(model.iList)
	offset_i_min = min(offset_i_min, min(model.iList{i}'));
end
for i = 1:length(model.iList)
	name = model.i_intervals{i}{1};

	if strcmp(name, 'R')
		list = model.iList{i}';
		r_x = [list(1:length(list)-1)-offset_i_min];
	end
end

% dg1_x = [0 delta+lambda*beta+alpha]
offset_dg1_min = inf;
for i = 1:length(model.bList)
	offset_dg1_min = min(offset_dg1_min, min(model.bList{i}'));
end
for i = 1:length(model.bList)
	name = model.b_intervals{i}{1};

	if strcmp(name, 'DG1')
		list = model.bList{i}';
		dg1_x = [list(1:length(list)-1)-offset_dg1_min];
	end
end


% cg1_x = [0 alpha+lambda*beta]
offset_cg1_min = inf;
for i = 1:length(model.tList)
	offset_cg1_min = min(offset_cg1_min, min(model.tList{i}'));
end
for i = 1:length(model.tList)
	name = model.t_intervals{i}{1};

	if strcmp(name, 'CG1')
		list = model.tList{i}';
		cg1_x = [list(1:length(list)-1)-offset_cg1_min];
	elseif strcmp(name, 'postG1')
		list = model.tList{i}';
		postg1_x = [list(1:length(list)-1)-offset_cg1_min];
	end
end

t_len = t_x(end)-t_x(1);
s_len = t_len * s_phase;
s_g2m_border = find(postg1_x <= cg1_x(end)+s_len, 1, 'last');
s_x = postg1_x(1:s_g2m_border);
g2m_x = postg1_x(s_g2m_border+1:end);

% b_y = DC
for i = 1:length(model.b_intervals)
	idx = model.b_intervals{i}{2};
	se = model.Hpos{idx};
	b_y_idx = [b_y_idx se(1):1:se(2)];
end

% t_y = C
for i = 1:length(model.t_intervals)
	idx = model.t_intervals{i}{2};
	se = model.Hpos{idx};
	t_y_idx = [t_y_idx se(1):1:se(2)];
end

% i_y = RC
for i=1:length(model.i_intervals)
	idx = model.i_intervals{i}{2};
	se = model.Hpos{idx};
	i_y_idx = [i_y_idx se(1):1:se(2)];
end

% r_y = R
for i=1:length(model.i_intervals)
	name = model.i_intervals{i}{1};

	if strcmp(name, 'R')
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		r_y_idx = [se(1):1:se(2)];
	end
end

% dg1_y = DG1
for i=1:length(model.b_intervals)
	name = model.b_intervals{i}{1};

	if strcmp(name, 'DG1')
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		dg1_y_idx = [se(1):1:se(2)];
	end
end

% cg1_y = CG1
for i=1:length(model.t_intervals)
	name = model.t_intervals{i}{1};

	if strcmp(name, 'CG1')
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		cg1_y_idx = [se(1):1:se(2)];
	elseif strcmp(name, 'postG1')
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		postg1_y_idx = [se(1):1:se(2)];
	end
end


s_y_idx = postg1_y_idx(1:s_g2m_border);
g2m_y_idx = postg1_y_idx(s_g2m_border+1:end);

% d_x / g1_x
% d_y_idx / g1_y_idx
border = max(dg1_x)-max(cg1_x);
pos = find(dg1_x>=border, 1);
d_x = dg1_x(1:pos-1);
g1_x = dg1_x(pos:end);
d_y_idx = dg1_y_idx(1:pos-1);
g1_y_idx = dg1_y_idx(pos:end);

model.t_x = t_x;
model.b_x = b_x;
model.i_x = i_x;
model.r_x = r_x;
model.dg1_x = dg1_x;
model.cg1_x = cg1_x;
model.d_x = d_x;
model.g1_x = g1_x;
model.s_x = s_x;
model.g2m_x = g2m_x;
model.postg1_x = postg1_x;

model.t_y_idx = t_y_idx;
model.b_y_idx = b_y_idx;
model.i_y_idx = i_y_idx;
model.r_y_idx = r_y_idx;
model.dg1_y_idx = dg1_y_idx;
model.cg1_y_idx = cg1_y_idx;
model.d_y_idx = d_y_idx;
model.g1_y_idx = g1_y_idx;
model.s_y_idx = s_y_idx;
model.g2m_y_idx = g2m_y_idx;
model.postg1_y_idx = postg1_y_idx;

return;
