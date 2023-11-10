
classdef DeconvolutionConfig
    methods (Static)

        function config = initializewithFixedConfigParameters()

            % Some fixed constants names, maybe put these elsewhere
            config.DECONV_MU0POS = 1;
            config.DECONV_LAMBDAPOS = 2;
            config.DECONV_DELTAPOS = 3;
            config.DECONV_SIGMA0POS = 4;
            config.DECONV_SIGMAVPOS = 5;
            config.DECONV_ALPHAPOS = 6;
            config.DECONV_BETAPOS = 7;

            config.DECONV_WAVELET = 1;
            config.DECONV_DIFF = 2;

            config.DECONV_WT1 = 'WT1';
            config.DECONV_WT2 = 'WT2';
            config.DECONV_JOINT = 'JOINT';

            % Parameter to set the type of deconvolution kernel,
            % There are other parameters, but we are just using 
            % wavelets for now...
            config.DECONV_KERNEL = config.DECONV_WAVELET;
        end

        % The Yulong Li Cell Cycle Experiment from 2017-2018. This
        % configuration is the gene expression just for replicate 2.
        % Note that replicate 2 is copied twice, TODO: is to allow for
        % single replicate runs too not just JOINT.
        function config = yl2_replicate2_rg1_gene_expression_config()

            config = DeconvolutionConfig.initializewithFixedConfigParameters();
           
            config.DECONV_DATASET = 'datasets/yl_cell_cycle/';

            config.DATA_WT1 = strcat(config.DECONV_DATASET, 'replicate2_gene_expression.txt');
            config.DATA_WT2 = strcat(config.DECONV_DATASET, 'replicate2_gene_expression.txt');
            
            config.GENESET_PATH = strcat(config.DECONV_DATASET, 'genes.lst');
            config.GENEMAPPING_PATH = 'datasets/yl_cell_cycle/gene_to_orf_name_mapping.txt';

            config.WT1_TP = [0 10 20 30 40 50 60 70 80 90 100 120 130 140];
            config.WT2_TP = [0 10 20 30 40 50 60 70 80 90 100 120 130 140];
            
            config.MODEL_WT1 = 'models/yl_cell_cycle/wt2_rg1.label';
            config.MODEL_WT2 = 'models/yl_cell_cycle/wt2_rg1.label';
        end

        % The gene expression data from Xin Guo's paper from 2011, 
        % using the Orlando 2007 dataset. Refer to 
        % http://deconvolution.cs.duke.edu/download.html
        function config = xg_gene_expression_config()
    
            config = DeconvolutionConfig.initializewithFixedConfigParameters();
           
            config.DECONV_DATASET = 'datasets/original_budflow/';
    
            config.DATA_WT1 = strcat(config.DECONV_DATASET, 'replicate1_gene_expression.txt');
            config.DATA_WT2 = strcat(config.DECONV_DATASET, 'replicate2_gene_expression.txt');
            
            config.GENESET_PATH = strcat(config.DECONV_DATASET, 'genes.lst');
            config.GENEMAPPING_PATH = 'datasets/original_budflow/gene_to_orf_name_mapping.txt';
    
            config.WT1_TP = 30:16:254;
            config.WT2_TP = 38:16:262;
    
            config.MODEL_WT1 = 'models/original_budflow/wt1_budflow/1.1.1.26.label';
            config.MODEL_WT2 = 'models/original_budflow/wt2_budflow/1.1.1.27.label';
        end

    end
end
