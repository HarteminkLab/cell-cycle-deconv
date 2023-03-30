classdef Model
	% A model class to deconvolve gene expression data from CLOCCS cell cycle
	% parameters

	properties
		% Define the properties of the class
		genename
		modeltype
		alpha
		orig_orfname
		orfname
		orfid
		datatype
		modelprefix
		g
		H
		timepoints
		lengths
		relations
		iList
		tList
		bList
		i_intervals
		t_intervals
		b_intervals
		Hsegments
		Hpos
	end

	methods

		function model = Model(genename)

			model.genename = genename;
			model.modelprefix = '1.1.1';
			model.modeltype = 'CDG1';
			model.alpha = [26 27];

			outdir = 'output/';
			modeldir1 = 'models/wt1_budflow/';
			modelfile1 = sprintf('%s.%d.label', model.modelprefix, model.alpha(1));

			modeldir2 = 'models/wt2_budflow/';
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

			modelprefix = model.modelprefix;

			dataset1 = load(Deconv.DATA_WT1, 'ascii');
			dataset2 = load(Deconv.DATA_WT2, 'ascii');

			g1 = dataset1(orfid,:)';
			g2 = dataset2(orfid,:)';
			model.g = [g1' g2'];
			model.H = [];
			
			% calculate H for WT1
			modelpath1 = strcat(modeldir1, modelfile1);

			model = model.loadModelFormat(modelpath1);
			model.timepoints = Deconv.WT1_TP;
			[H, Hsegments, Hpos] = calcH(model);
			model.H = H;
			model.Hsegments = Hsegments;
			model.Hpos = Hpos;

			% calculate H for WT2
			modelpath2 = strcat(modeldir2, modelfile2);
			model = model.loadModelFormat(modelpath2);
			model.timepoints = [Deconv.WT1_TP Deconv.WT2_TP];
			[H, Hsegments, Hpos] = calcH(model);
			model.H = H;
		end

		% Method 1
		function ret = loadModelFormat(obj, modelpath)

			[lengths, relations, iList, tList, bList, i_intervals, t_intervals, b_intervals] = readModelFormat(modelpath, obj);

			obj.lengths = lengths;
			obj.relations = relations;
			obj.iList = iList;
			obj.tList = tList;
			obj.bList = bList;
			obj.i_intervals = i_intervals;
			obj.t_intervals = t_intervals;
			obj.b_intervals = b_intervals;

			ret = obj;
		end

	 end
end
