
classdef Model
	% A model class to deconvolve gene expression data from CLOCCS cell cycle
	% parameters

	properties
		% Define the properties of the class
		genename, modeltype
		orig_orfname
		orfname
		orfid
		datatype
		modelprefix
		% residual norm measures the accuracy of the fit
		% solution norm measures the weighting of the smoothing wavelet portion of the solution
		base_rn, rn, sn
		g, g1, g2, timepoints
		timepoints1, timepoints2
		intervals
		H, Hsegments, Hpos,
		gm, f, pred_g
		pred_g1, pred_g2
		f_initial, f_top, f_bottom
		f_initial_list, f_top_list, f_bottom_list
		massesTime
	end

	methods

		function model = Model(genename, gamma)

			model.gm = gamma;
			model.genename = genename;
			model.modelprefix = '1.1.1';
			model.modeltype = 'CDG1';

			outdir = 'output/';

			orfname = gene_to_orfname(genename, Deconv.NAME_MAPPING);

			% deal with orfname and datatype
			orig_orfname = orfname;
			[orfname, orfid] = map2SystemNames(orig_orfname);

			model.orig_orfname = orig_orfname;
			model.orfname = orfname;
			model.genename = genename;
			model.orfid = orfid;
			model.datatype = Deconv.DECONV_JOINT;

			modelprefix = model.modelprefix;

			dataset1 = load(Deconv.DATA_WT1, 'ascii');
			dataset2 = load(Deconv.DATA_WT2, 'ascii');

			g1 = dataset1(orfid,:)';
			g2 = dataset2(orfid,:)';
			model.g = [g1' g2'];
			model.H = [];

			% calculate H for WT1
			model.intervals = ModelIntervals(Deconv.MODEL_WT1, model);
			model.timepoints = Deconv.WT1_TP;
			[H1, Hsegments, Hpos] = calcH(model);
			model.Hsegments = Hsegments;
			model.Hpos = Hpos;

			% calculate H for WT2 and combine into a joint H
			model.intervals = ModelIntervals(Deconv.MODEL_WT2, model);
			model.timepoints = Deconv.WT2_TP;
			[H2, Hsegments, Hpos] = calcH(model);

			% Merge the H kernels
			model.timepoints = [Deconv.WT1_TP Deconv.WT2_TP];
			model.timepoints1 = Deconv.WT1_TP;
			model.timepoints2 = Deconv.WT2_TP;
			model.g1 = g1;
			model.g2 = g2;

			model.H = [H1' H2']';
		end
	 end
end
