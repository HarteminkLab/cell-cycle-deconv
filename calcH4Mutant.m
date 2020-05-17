function [model, H] = calcH4Mutant(model)

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
%global DECONV_DELTAPOS;
%global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;

% -------------------------------------------
% -------------------------------------------

mu0 = model.lengths(DECONV_MU0POS);
lambda = model.lengths(DECONV_LAMBDAPOS);
%delta = model.lengths(DECONV_DELTAPOS);
%alpha = model.lengths(DECONV_ALPHAPOS);
sigma0 = model.lengths(DECONV_SIGMA0POS);
sigmav = model.lengths(DECONV_SIGMAVPOS);

SCALING = 1;

INIT2ZERO = 0;

max_R = 10;  % runs

%disp(sprintf('mu0=%d\nlambda=%d\nsigma0=%f\nsigmav=%f\n', mu0, lambda, sigma0, sigmav));

if length(model.timepoints) == 2 % two replicates
	timepoints = model.timepoints{2};
else
	timepoints = model.timepoints{1};
end

num_timepoints = size(timepoints, 2);

% initial branch
iDs = {};
for i=1:length(model.iList)
	iDs{i} = zeros(num_timepoints, length(model.iList{i})-1);
end

% top branch
tDs = {};
for i=1:length(model.tList)
	tDs{i} = zeros(num_timepoints, length(model.tList{i})-1);
end

%disp(sprintf('Generating convolution matrix H\n'));

% ----------------------------------------------
% ----------------------------------------------
for i = 1:num_timepoints
	t = timepoints(i);

	% --------------------------------------------------
	% {0,0} cohort
	% initial branch
	for iList_idx = 1:length(model.iList)
		array = model.iList{iList_idx};
		cdf = normcdf(array, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
		iDs{iList_idx}(i,:) = (cdf(2:length(array))-cdf(1:(length(array)-1)));
	end

	% top branch (from run 1)
	for runs = 1:max_R
		for tList_idx = 1:length(model.tList)
			array = model.tList{tList_idx};
			cdf = normcdf(array+runs*lambda, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
			tDs{tList_idx}(i,:) = tDs{tList_idx}(i,:)+(cdf(2:length(array))-cdf(1:length(array)-1));
		end
	end

	% --------------------------------------------------

	if INIT2ZERO
		array_sum = 0;

		for iList_idx = 1:length(model.iList);
			array_sum = array_sum + sum(iDs{iList_idx}(i,:));
		end
		for tList_idx = 1:length(model.tList);
			array_sum = array_sum + sum(tDs{tList_idx}(i,:));
		end

		array_sum
		iDs{1}(i,1) = iDs{1}(i,1) + (1-array_sum);
	end

end 

Hsegments = {};
for i = 1:length(model.relations)
	relation = model.relations{i};
	name = relation{1};

	for idx = 2:2:length(relation)-1
		label = relation{idx};
		num = str2num(relation{idx+1})+1;
		if strcmp(label, 'i')
			matrix = iDs{num};
		elseif strcmp(label, 't')
			matrix = tDs{num};
		end
		if idx == 2
			Hsegments{i} = matrix;
		else
			Hsegments{i} = Hsegments{i}+matrix;
		end
	end
end

model.Hsegments = Hsegments;

H = [];
Hpos = {};
cur_start = 1;
for i=1:length(Hsegments)
	cur_len = size(Hsegments{i}, 2);
	cur_end = cur_start + cur_len - 1;
	Hpos{i} = [cur_start cur_end];
	cur_start = cur_end+1;
	H = [H Hsegments{i}];
end

if SCALING
	for i=1:size(H,1)
		w = sum(H(i,:));
		H(i,:) = H(i,:)./w;
	end
end


if isfield(model, 'H')
	model.H = [model.H' H']';
else
	model.H = H;
end

model.Hpos = Hpos;
