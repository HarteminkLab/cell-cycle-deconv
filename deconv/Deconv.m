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

		DECONV_WT1 = 'WT1';
		DECONV_WT2 = 'WT2';
		DECONV_JOINT = 'JOINT';

		% ===========================  YL  ==========================================

		MODEL_WT1 = 'models/yl_cell_cycle/wt1.label';
		MODEL_WT2 = 'models/yl_cell_cycle/wt2.label';

		DECONV_DATASET = 'datasets/yl_cell_cycle/';
		NAME_MAPPING = 'datasets/yl_cell_cycle/map2sys2.txt';

		DATA_WT1 = strcat(Deconv.DECONV_DATASET, 'wt1.txt');
		DATA_WT2 = strcat(Deconv.DECONV_DATASET, 'wt2.txt');

		WT1_TP = [0	20	30	40	50	60	70	80	90	100	110	120	130	140	150];
		WT2_TP = [0	10	20	30	40	50	60	70	80	90	100	120	130	140];

		% ======================    Original    ========================================

		% MODEL_WT1 = 'models/original_budflow/wt1_budflow/1.1.1.26.label';
		% MODEL_WT2 = 'models/original_budflow/wt2_budflow/1.1.1.27.label';

		%DECONV_DATASET = 'datasets/original_budflow/';
		%NAME_MAPPING = 'datasets/original_budflow/map2sys2.txt';

		%DATA_WT1 = strcat(Deconv.DECONV_DATASET, 'wt1.txt');
		%DATA_WT2 = strcat(Deconv.DECONV_DATASET, 'wt2.txt');

		%WT1_TP = 30:16:254;
		%WT2_TP = 38:16:262;
	end
end
