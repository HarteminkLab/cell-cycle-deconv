function [model] = test_deconv(orfname, modeltype, datatype, alpha, gamma, learn_flag, fig_flag)

global SLIENCE;

global DECONV_MU0POS;			% mu0
global DECONV_LAMBDAPOS;	% lambda
global DECONV_DELTAPOS;		% delta
global DECONV_ALPHAPOS;		% alpha
global DECONV_SIGMA0POS;	% sigma_0
global DECONV_SIGMAVPOS;	% sigma_v
global DECONV_BETAPOS;		% beta

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

global FROM_FINDOPTIMAL;

SLIENCE = 1;

model = deconvolve(orfname, modeltype, datatype, alpha, 0, 0, 0);
base_rn = model.rn;

model = deconvolve(orfname, modeltype, datatype, alpha, gamma, 0, 1);
rn = model.rn;

a = rn - base_rn;
r = (rn-base_rn)/base_rn*100;

disp(sprintf('rate = %0.3g%%, base = %0.3g, diff = %0.3g', r, base_rn, a));
