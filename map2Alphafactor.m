function [id]  = map2Alphafactor(namelist, flag)

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

if exist('flag', 'var') && flag == 2
	if exist(namelist, 'file')
		[namelist] = textread(namelist, '%s');
	else
		error(sprintf('Not exist file %s', namelist));
	end
end

[names] = textread(strcat(DECONV_DATASET, 'af_names.lst'), '%s');

id = [];

if iscellstr(namelist)
	for idx=1:length(namelist)
		name = namelist{idx};
		pos = strmatch(name, names, 'exact');

		n = length(pos);
		if n == 0;
			error(sprintf('Unknown identifier %s', name));
		elseif n > 1;
			error(sprintf('Ambiguous identifier %s', name));
		else
			id(idx) = pos;
		end
	end
else % just one name string
	name = namelist;
	pos = strmatch(name, names, 'exact');

	n = length(pos);
	if n == 0;
		error(sprintf('Unknown identifier %s', name));
	elseif n > 1;
		error(sprintf('Ambiguous identifier %s', name));
	else
		id(1) = pos;
	end
end

return;
