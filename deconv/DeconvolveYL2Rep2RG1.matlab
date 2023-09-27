% Configuration for Yulong 2018 Replicate 2 Dataset (TODO: Currently duplicating replicate 2 for WT1 and WT2)
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
		DECONV_DATASET = 'datasets/yl_cell_cycle/';

		DECONV_WT1 = 'WT1';
		DECONV_WT2 = 'WT2';
		DECONV_JOINT = 'JOINT';

		DATA_WT1 = strcat(Deconv.DECONV_DATASET, 'replicate2_gene_expression.txt');
		DATA_WT2 = strcat(Deconv.DECONV_DATASET, 'replicate2_gene_expression.txt');

		WT1_TP = [0	10	20	30	40	50	60	70	80	90	100	120	130	140];
		WT2_TP = [0	10	20	30	40	50	60	70	80	90	100	120	130	140];
		
		MODEL_WT1 = 'models/yl_cell_cycle/wt2_rg1.model';
        MODEL_WT2 = 'models/yl_cell_cycle/wt2_rg1.model';
	end
end
