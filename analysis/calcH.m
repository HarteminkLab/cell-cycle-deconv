function [model, H] = calcH(model)

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_DELTAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;

mu0 = model.lengths(DECONV_MU0POS);
lambda = model.lengths(DECONV_LAMBDAPOS);
delta = model.lengths(DECONV_DELTAPOS);
alpha = model.lengths(DECONV_ALPHAPOS);
sigma0 = model.lengths(DECONV_SIGMA0POS);
sigmav = model.lengths(DECONV_SIGMAVPOS);

SCALING = 1;

max_cellcycles = 10;
max_R = 10;
max_G = 10;

if size(model.timepoints, 1) == 2
	timepoints = model.timepoints(2,:);
else
	timepoints = model.timepoints;
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

% bottom branch
bDs = {};
for i=1:length(model.bList)
	bDs{i} = zeros(num_timepoints, length(model.bList{i})-1);
end

% ----------------------------------------------
% ----------------------------------------------
for i = 1:num_timepoints
	% Q(t): total size of cohorts at time t
	t = timepoints(i);
	Q = 0;
	for (r = 0:max_R)
		Q = Q+Qr(mu0,sigma0,sigmav,delta,lambda,t,r,alpha);
	end

	% --------------------------------------------------
	% fraction of {0,0} cohort = Q(R,t)/Q(t)
	frac_init = Qr(mu0,sigma0,sigmav,delta,lambda,t,0,alpha)/Q;

	% {0,0} cohort
	% initial branch
	for iList_idx = 1:length(model.iList)
		array = model.iList{iList_idx};
		cdf = normcdf(array, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
		iDs{iList_idx}(i,:) = (cdf(2:length(array))-cdf(1:(length(array)-1)))*frac_init;
	end
	% top branch (from run 1)
	for runs = 1:max_R
		for tList_idx = 1:length(model.tList)
			array = model.tList{tList_idx};
			cdf = normcdf(array+runs*lambda, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
			tDs{tList_idx}(i,:) = tDs{tList_idx}(i,:)+(cdf(2:length(array))-cdf(1:length(array)-1))*frac_init;
		end
	end

	% all other cohorts / exclude {0,0}
	frac_rest_all = 0;
	for r = 1:max_R
		for g = 1:r
			% M(g,r,t): size of all cohorts {g,r} at time t
			% fraction = M(g,r,t) / Q(t)
			frac_rest = Mgr(mu0,sigma0,sigmav,delta,lambda,t,g,r,alpha)/Q;
			frac_rest_all = frac_rest_all+frac_rest;

			if frac_rest > 1e-10
				% top branch
				trun_cdf = normcdf(r*lambda+(g-1)*delta-alpha, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
				trun_denom = 1-trun_cdf;

				for tList_idx = 1:length(model.tList)
					array = model.tList{tList_idx};
					% note: run r \in b
					for runs = r+1:max_R
						cdf = normcdf(array+runs*lambda+g*delta, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
						if trun_denom == 0
							portion = cdf*0;
						else
							portion = (cdf-trun_cdf)/trun_denom;
						end

						tDs{tList_idx}(i,:) = tDs{tList_idx}(i,:)+(portion(2:length(array))-portion(1:length(array)-1))*frac_rest;
					end
				end

				% bottom branch
				for bList_idx = 1:length(model.bList)
					array = model.bList{bList_idx};
					cdf = normcdf(array+r*lambda+g*delta, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
					if trun_denom == 0
						portion = cdf*0;
					else
						portion = (cdf-trun_cdf)/trun_denom;
					end
					portion(portion<0) = 0;

					bDs{bList_idx}(i,:) = bDs{bList_idx}(i,:)+(portion(2:length(array))-portion(1:length(array)-1))*frac_rest;
				end
			end % frac_rest > 0
		end % end R
	end % end G

	% --------------------------------------------------
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
		elseif strcmp(label, 'b')
			matrix = bDs{num};
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

% Joint model, so WT2 H is joined with the existing H computed for WT1
if isfield(model, 'H')
	model.H = [model.H' H']';
else
	model.H = H;
end

model.Hpos = Hpos;

% ----------------------------------------------------------------
% ----------------------------------------------------------------

function[N] = Qr(mu0,sigma0,sigmav,delta,lambda,t,r,alpha)

% Q(r,t): all cohorts with reproductive instance r at time t relative to size of {0,0} cohort

START=1000;
if(r == 0)
	N = START;
%	normval = 1-normcdf(-mu0, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
%	N = normval*START;
else
	N = 0;
  for i = 0:r-1
		normval = 1-normcdf(r*lambda+i*delta-alpha, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
		N = N+normval*START*nchoosek(r-1,i);
	end
end

return;

% ----------------------------------------------------------------

function[N] = Mgr(mu0,sigma0,sigmav,delta,lambda,t,g,r,alpha)
% M(g,r,t): size of cohort(g,r) at time t relative to size of {0,0} cohort

START=1000;
N=0;
if(g == 0)
	if(r == 0)
		N = START;
	else
		N = 0;
	end
elseif(g > 0)
	if(r < g)
		N = 0;
	else
		normval = 1-normcdf(r*lambda+(g-1)*delta-alpha, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
		N = normval*START*nchoosek(r-1,g-1);
	end
else
	disp('Error : g < 0');
end
