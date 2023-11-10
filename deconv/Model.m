    classdef Model
	% A model class to deconvolve gene expression data from CLOCCS cell cycle
	% parameters

	properties
		% Define the properties of the class
		genename, modeltype
		alpha, orig_orfname
		orfname
		orfid
		datatype
		modelprefix
		g, timepoints
        rn, sn % The residual and smoothing norms
        base_rn % For finding the optimal gamma value
		intervals
		H, Hsegments, Hpos,
		gm, f, pred_g
		f_initial, f_top, f_bottom
		f_initial_list, f_top_list, f_bottom_list

        % Let's store some of the deconvolution data objects for debuggin
        W1, W2, mean_g, f_padded,

        % For finding optimal gamma
        err, rn0, sn_nogamma

        % A struct containing configuration parameters like the path
        % to data and model files
        config,
	end

	methods

        function model = Model(config, genename, gamma)

            model.config = config;
			model.gm = gamma;
			model.genename = genename;
			orfname = gene_to_orfname(config, genename);

			% deal with orfname and datatype
			orig_orfname = orfname;
			[orfname, orfid] = map2SystemNames(config, orfname);

			model.orig_orfname = orig_orfname;
			model.orfname = orfname;
			model.genename = genename;
			model.orfid = orfid;
			model.datatype = config.DECONV_JOINT;

			dataset1 = load(config.DATA_WT1, 'ascii');
			dataset2 = load(config.DATA_WT2, 'ascii');

			g1 = dataset1(orfid,:)';
			g2 = dataset2(orfid,:)';
			model.g = [g1' g2'];
			model.H = [];

            % Error handling for when a gene isn't loaded properly
            if (size(model.g, 1) == 0) 
                error("g should have at least 1 row. Check that g was loaded correctly for the given gene.")
            end
			
			% calculate H for WT1
			modelpath1 = config.MODEL_WT1;

			model.intervals = ModelIntervals(config, modelpath1, model);
			model.timepoints = config.WT1_TP;
			[H1, Hsegments, Hpos] = calcH(config, model);
			model.Hsegments = Hsegments;
			model.Hpos = Hpos;

			% calculate H for WT2 and combine into a joint H
			modelpath2 = config.MODEL_WT2;
			model.intervals = ModelIntervals(config, modelpath2, model);
			model.timepoints = config.WT2_TP;
			[H2, Hsegments, Hpos] = calcH(config, model);

			% Merge the H kernels
			model.timepoints = [config.WT1_TP' config.WT2_TP']';
			model.H = [H1' H2']';
		end
	 end
end
