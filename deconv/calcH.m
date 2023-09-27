function [H, Hsegments, Hpos] = calcH(model)

	intervals = model.intervals;
	parameters = intervals.getCellCycleParameters();

	mu0 = parameters(Deconv.DECONV_MU0POS);
	lambda = parameters(Deconv.DECONV_LAMBDAPOS);
	delta = parameters(Deconv.DECONV_DELTAPOS);
	alpha = parameters(Deconv.DECONV_ALPHAPOS);
	sigma0 = parameters(Deconv.DECONV_SIGMA0POS);
	sigmav = parameters(Deconv.DECONV_SIGMAVPOS);

	max_cellcycles = 10;
	max_R = 10;
	max_G = 10;

	timepoints = model.timepoints;
	num_timepoints = size(timepoints, 2);

	initialTimepointsList = intervals.initialTimepointsList;
	bottomTimepointsList = intervals.bottomTimepointsList;
	topTimepointsList = intervals.topTimepointsList;

	% For each timepoints vector, create a matrix with the number of timepoints as rows
	% and the timepoint intervals as columns 1 per vector of timepoints


	% For the initial timepoint intervals
	initialBranchPartialH = {};
	for i=1:length(initialTimepointsList)
		initialBranchPartialH{i} = zeros(num_timepoints, length(initialTimepointsList{i})-1);
	end

	% The top intervals
	topBranchPartialH = {};
	for i=1:length(topTimepointsList)
		topBranchPartialH{i} = zeros(num_timepoints, length(topTimepointsList{i})-1);
	end

	% The bottom intervals
	bottomBranchPartialH = {};
	for i=1:length(bottomTimepointsList)
		bottomBranchPartialH{i} = zeros(num_timepoints, length(bottomTimepointsList{i})-1);
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
		for initialTimepointsList_idx = 1:length(initialTimepointsList)
			array = initialTimepointsList{initialTimepointsList_idx};
			cdf = normcdf(array, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
			initialBranchPartialH{initialTimepointsList_idx}(i,:) = (cdf(2:length(array))-cdf(1:(length(array)-1)))*frac_init;
		end
		% top branch (from run 1)
		for runs = 1:max_R
			for topTimepointsList_idx = 1:length(topTimepointsList)
				array = topTimepointsList{topTimepointsList_idx};
				cdf = normcdf(array+runs*lambda, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
				topBranchPartialH{topTimepointsList_idx}(i,:) = topBranchPartialH{topTimepointsList_idx}(i,:)+(cdf(2:length(array))-cdf(1:length(array)-1))*frac_init;
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

					for topTimepointsList_idx = 1:length(topTimepointsList)
						array = topTimepointsList{topTimepointsList_idx};
						% note: run r \in b
						for runs = r+1:max_R
							cdf = normcdf(array+runs*lambda+g*delta, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
							if trun_denom == 0
								portion = cdf*0;
							else
								portion = (cdf-trun_cdf)/trun_denom;
							end

							topBranchPartialH{topTimepointsList_idx}(i,:) = topBranchPartialH{topTimepointsList_idx}(i,:)+(portion(2:length(array))-portion(1:length(array)-1))*frac_rest;
						end
					end

					% bottom branch
					for bottomTimepointsList_idx = 1:length(bottomTimepointsList)
						array = bottomTimepointsList{bottomTimepointsList_idx};
						cdf = normcdf(array+r*lambda+g*delta, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
						if trun_denom == 0
							portion = cdf*0;
						else
							portion = (cdf-trun_cdf)/trun_denom;
						end
						portion(portion<0) = 0;

						bottomBranchPartialH{bottomTimepointsList_idx}(i,:) = bottomBranchPartialH{bottomTimepointsList_idx}(i,:)+(portion(2:length(array))-portion(1:length(array)-1))*frac_rest;
					end
				end % frac_rest > 0
			end % end R
		end % end G

		% --------------------------------------------------
	end 

	Hsegments = {};
	relations = intervals.relations;

	for i = 1:length(relations)
		relation = relations{i};
		name = relation{1};
		for idx = 2:2:length(relation)-1
			label = relation{idx};
			num = str2num(relation{idx+1})+1;
			if strcmp(label, 'i')
				matrix = initialBranchPartialH{num};
			elseif strcmp(label, 't')
				matrix = topBranchPartialH{num};
			elseif strcmp(label, 'b')
				matrix = bottomBranchPartialH{num};
			end

			if idx == 2
				Hsegments{i} = matrix;
			else
				Hsegments{i} = Hsegments{i}+matrix;
	        end
		end
	end

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

	% Scale the final matrix such that each row has an equal sum
	for i=1:size(H,1)
		w = sum(H(i,:));
		H(i,:) = H(i,:)./w;
	end
end

% ----------------------------------------------------------------
% ----------------------------------------------------------------

function[N] = Qr(mu0,sigma0,sigmav,delta,lambda,t,r,alpha)
	% Q(r,t): all cohorts with reproductive instance r at time t relative to size of {0,0} cohort

	START=1000;
	if(r == 0)
		N = START;
	else
		N = 0;
	  for i = 0:r-1
			normval = 1-normcdf(r*lambda+i*delta-alpha, t-mu0, sqrt(sigma0^2+t^2*sigmav^2));
			N = N+normval*START*nchoosek(r-1,i);
		end
	end
end

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
end
