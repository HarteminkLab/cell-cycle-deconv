function [lengths, relations, iList, tList, bList, i_intervals, t_intervals, b_intervals] = readModelFormat(config, modelfile, model)

	lengths = zeros(1,6);

	if exist(modelfile, 'file') == 0
		error('The model file %s does not exist');
	end

	% headers
	LENGTHS = '# lengths';
	DESCRIPTION = '# description';
	I = '# i';
	T = '# t';
	B = '# b';

	iList_idx = 1;
	iList = {};
	tList_idx = 1;
	tList = {};
	bList_idx = 1;
	bList = {};
	relations_idx = 1;
	relations = {};
	parseFlag = -1;

	fid = fopen(modelfile);
	while 1
		tline = fgetl(fid);

		if ~ischar(tline)
			break;
		elseif strcmp(tline, LENGTHS)
			parseFlag = 1;
		elseif strcmp(tline, DESCRIPTION)
			parseFlag = 2;
		elseif strcmp(tline, I)
			parseFlag = 3;
		elseif strcmp(tline, T)
			parseFlag = 4;
		elseif strcmp(tline, B)
			parseFlag = 5;
		% lengths
		elseif parseFlag == 1 
			[flag, value, pos] = parseLengths(config, tline);
			if flag == 0
				return;
			else
				lengths(pos) = value;
			end
		% description
		elseif parseFlag == 2
			relation = parseDescription(tline);
			relations{relations_idx} = relation;
			relations_idx = relations_idx+1;
		% interval i
		elseif parseFlag == 3
			interval = parseIntervals(tline);
			iList{iList_idx} = interval;
			iList_idx = iList_idx+1;
		% interval t
		elseif parseFlag == 4
			interval = parseIntervals(tline);
			tList{tList_idx} = interval;
			tList_idx = tList_idx+1;
		% interval b
		elseif parseFlag == 5
			interval = parseIntervals(tline);
			bList{bList_idx} = interval;
			bList_idx = bList_idx+1;
		elseif parseFlag == -1
			disp(sprintf('Error in line %s ... exiting', tline));
			flag = 0;
			return;
		end
	end
	fclose(fid);

	i_intervals = {};
	t_intervals = {};
	b_intervals = {};
	for i = 1:length(relations)
		relation = relations{i};
		notation = relation{1};

		for idx = 2:2:length(relation)-1
			label = relation{idx};
			num = str2num(relation{idx+1})+1;

			if label == 'i'
				i_intervals{num} = {notation, i};
			elseif label == 't'
				t_intervals{num} = {notation, i};
			elseif label == 'b'
				b_intervals{num} = {notation, i};
			end
		end
	end

return;

% ----------------------------------------
% ----------------------------------------

function [flag, value, pos] = parseLengths(config, tline)

	segments = strsplit(tline, ' ');
	if strcmp(segments{1}, 'mu0')
		pos = config.DECONV_MU0POS;
		value = str2num(segments{2});
		flag = 1;
	elseif strcmp(segments{1}, 'lambda')
		pos = config.DECONV_LAMBDAPOS;
		value = str2num(segments{2});
		flag = 1;
	elseif strcmp(segments{1}, 'delta')
		pos = config.DECONV_DELTAPOS;
		value = str2num(segments{2});
		flag = 1;
	elseif strcmp(segments{1}, 'alpha')
		pos = config.DECONV_ALPHAPOS;
		value = str2num(segments{2});
		flag = 1;
	elseif strcmp(segments{1}, 'sigma0')
		pos = config.DECONV_SIGMA0POS;
		value = str2num(segments{2});
		flag = 1;
	elseif strcmp(segments{1}, 'sigmav')
		pos = config.DECONV_SIGMAVPOS;
		value = str2num(segments{2});
		flag = 1;
	elseif strcmp(segments{1}, 'beta')
		pos = config.DECONV_BETAPOS;
		value = str2num(segments{2});
		flag = 1;
	else
		disp(sprintf('Error in line %s ... exiting', tline));
		flag = 0;
		value = 0;
		pos = -1;
	end
return;

% ----------------------------------------

function [relation] = parseDescription(tline)
	relation = strsplit(tline, ' ');
return;

% ----------------------------------------

function [interval] = parseIntervals(tline)
	segments = strsplit(tline, ' ');
	interval = [];
	for i=1:length(segments)
		interval(i) = str2num(segments{i});
	end
return;
