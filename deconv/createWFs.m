
function [wf_i, wf_t, wf_b] = createWFs(model)

	i_intervals = model.intervals.initialPhaseMapping;
	iList = model.intervals.initialTimepointsList;
	
	t_intervals = model.intervals.topPhaseMapping;
	tList = model.intervals.topTimepointsList;

	b_intervals = model.intervals.bottomPhaseMapping;
	bList = model.intervals.bottomTimepointsList;

	wf_i = createWeightF(model, i_intervals, iList);
	wf_t = createWeightF(model, t_intervals, tList);
	wf_b = createWeightF(model, b_intervals, bList);
end


function [wf] = createWeightF(model, intervals, timpointsList)

	wf = [];
	intervals = model.intervals.initialPhaseMapping;
	timepointsList = model.intervals.initialTimepointsList;

	for i = 1:length(intervals)
		idx = intervals{i}{2};
		se = model.Hpos{idx};
		gap = timepointsList{i}(2)-timepointsList{i}(1);
		wf = [wf ones(1, se(2)-se(1)+1)*gap];
	end
end
