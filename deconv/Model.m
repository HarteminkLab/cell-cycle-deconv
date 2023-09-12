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
		intervals
		H, Hsegments, Hpos,
		gm, f, pred_g
		f_initial, f_top, f_bottom
		f_initial_list, f_top_list, f_bottom_list
	end

	methods

		function model = Model(genename, gamma)

			model.gm = gamma;
			model.genename = genename;
			model.modelprefix = '1.1.1';
			model.modeltype = 'CDG1';
			model.alpha = [26 27];

			modeldir1 = 'models/original_budflow/wt1_budflow/';
			modelfile1 = sprintf('%s.%d.label', model.modelprefix, model.alpha(1));

			modeldir2 = 'models/original_budflow/wt2_budflow/';
			modelfile2 = sprintf('%s.%d.label', model.modelprefix, model.alpha(2));

			orfname = gene_to_orfname(genename);

			% deal with orfname and datatype
			orig_orfname = orfname;
			[orfname, orfid] = map2SystemNames(orig_orfname);

			model.orig_orfname = orig_orfname;
			model.orfname = orfname;
			model.genename = genename;
			model.orfid = orfid;
			model.datatype = Deconv.DECONV_JOINT;

			dataset1 = load(Deconv.DATA_WT1, 'ascii');
			dataset2 = load(Deconv.DATA_WT2, 'ascii');

			g1 = dataset1(orfid,:)';
			g2 = dataset2(orfid,:)';
			model.g = [g1' g2'];
			model.H = [];

            % Error handling for when a gene isn't loaded properly
            if (size(model.g, 1) == 0) 
                error("g should have at least 1 row. Check that g was loaded correctly for the given gene.")
            end
			
			% calculate H for WT1
			modelpath1 = strcat(modeldir1, modelfile1);
			model.intervals = ModelIntervals(modelpath1, model);
			model.timepoints = Deconv.WT1_TP;
			[H1, Hsegments, Hpos] = calcH(model);
			model.Hsegments = Hsegments;
			model.Hpos = Hpos;

			% calculate H for WT2 and combine into a joint H
			modelpath2 = strcat(modeldir2, modelfile2);
			model.intervals = ModelIntervals(modelpath2, model);
			model.timepoints = Deconv.WT2_TP;
			[H2, Hsegments, Hpos] = calcH(model);

			% Merge the H kernels
			model.timepoints = [Deconv.WT1_TP' Deconv.WT2_TP']';
			model.H = [H1' H2']';
		end
	 end
end
