classdef Deconv
	properties (Constant)
		DECONV_MU0POS = 1;
		DECONV_LAMBDAPOS = 2;
		DECONV_DELTAPOS = 3;
		DECONV_SIGMA0POS = 4;
		DECONV_SIGMAVPOS = 5;
		DECONV_ALPHAPOS = 6;
		DECONV_BETAPOS = 7;

		DECONV_WAVELET = 1;
		DECONV_DIFF = 2;

		DECONV_KERNEL = Deconv.DECONV_WAVELET;
		DECONV_DATASET = 'datasets/original_budflow/';

		DECONV_WT1 = 'WT1';
		DECONV_WT2 = 'WT2';
		DECONV_JOINT = 'JOINT';

		DATA_WT1 = strcat(Deconv.DECONV_DATASET, 'wt1.txt');
		DATA_WT2 = strcat(Deconv.DECONV_DATASET, 'wt2.txt');

		WT1_TP = 30:16:254;
		WT2_TP = 38:16:262;

		MODEL_DIR = 'models/';
	end
end
