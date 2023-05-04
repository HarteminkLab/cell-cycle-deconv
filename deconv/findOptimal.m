function [model, rns, gammas, lruns, rruns] = findOptimal(model)

ELBOW_BINS = 10;

SMALL = 5e-5;
FROM_FINDOPTIMAL = 1;

% ======================
% some settings

DEFAULT_RN_CUTOFF = 10;     % base_rn is too_large
DEFAULT_GM = 0.004;         % default gamma if base_rn is larger than cutoff

% gamma boundary
GAMMA_MIN = 0.001;
GAMMA_MAX = 0.01;

% left boundary
rn_rate_left = 1.10;
left_rn = 0.08;


% right boundary
rn_rate_right = 1.40;
right_rn = 0.32;

% End of the settings
% ======================

% find best fit
model.gm = 0;
model = deconvolve(model);
base_rn = model.rn;
model.base_rn = base_rn;

flag = 1; % not using the default_gm
if base_rn >= DEFAULT_RN_CUTOFF
    model.gm = DEFAULT_GM;
    flag = 0;
end

% left boundary search
if flag
    gm_left = GAMMA_MIN;
    gm_right = GAMMA_MAX;

    rn_left = min(rn_rate_left*base_rn, base_rn+left_rn);
    leftr = (rn_left/base_rn-1)*100;
    model.gm = gm_left;
    [model] = deconvolve(model);

    if model.rn >= DEFAULT_RN_CUTOFF;
        model.gm = DEFAULT_GM;
        flag = 0;
    end
end

if flag
    bs_flag = 1;
    LR_SMALL = 0;
    lruns = 0;
    if model.rn < rn_left
        [model, lruns, bs_flag] = binarysearch(model, gm_left, gm_right, rn_left);
    end

    if bs_flag == 0
        flag = 0;
        model.gm = DEFAULT_GM;
    else
        gm_left = model.gm;
        rn_left = model.rn;
    end
end

% right boundary search
if flag
    rn_right = max(rn_rate_right*base_rn, base_rn+right_rn);
    rightr = (rn_right/base_rn-1)*100;
    model.gm = gm_right;
    [model] = deconvolve(model);
    if model.rn >= DEFAULT_RN_CUTOFF;
        model.gm = DEFAULT_GM;
        flag = 0;
    end
end

if flag
    bs_flag = 1;
    rruns = 0;
    if model.rn > rn_right
        [model, rruns, bs_flag] = binarysearch(model, gm_left, gm_right, rn_right);
    end

    if bs_flag == 0
        flag = 0;
        model.gm = DEFAULT_GM;
    else
        gm_right = model.gm;
        rn_right = model.rn;
    end
end

if flag
    bs_flag = 1;
    if (abs(gm_right-gm_left) < SMALL) % gm_right == gm_left
        model.gm = (gm_right+gm_left)/2;
    else
        step = (gm_right-gm_left)/ELBOW_BINS;
        gamma_array = gm_left:step:gm_right;
        [model.gm, bs_flag, rns, gammas] = findElbow(model, gamma_array);
    end

    if bs_flag == 0
        flag = 0;
        model.gm = DEFAULT_GM;
    end
end

disp(sprintf('%s: ... final gamma = %0.5f\n', model.orig_orfname, model.gm));
[model] = deconvolve(model);

return;

% ===============================

function [model, runs, flag] = binarysearch(model, gamma_min, gamma_max, rn_goal)
global DEFAULT_RN_CUTOFF;

    RN_SMALL = 2e-4;
    LR_SMALL = 5e-4;
    flag = 1;

    left = gamma_min;
    right = gamma_max;
    runs = 0;

    % find the fit_left point
    while right-left > LR_SMALL
        cur_gamma = (left+right)/2;
        runs = runs+1;
        
        model.gm = cur_gamma;
        
        [model] = deconvolve(model);

        if model.rn >= DEFAULT_RN_CUTOFF
            flag = 0;
            return;
        end

        if abs(model.rn-rn_goal) <= RN_SMALL
            break;
        elseif model.rn > rn_goal
            right = cur_gamma;
        else
            left = cur_gamma;
        end

        rn_rate = (model.rn/model.base_rn-1)*100;

    end
    return;
