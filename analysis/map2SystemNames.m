function [syslist, sysids, namelist] = map2SystemNames(namelist, flag)

% flag = 1 or undefined:
%		namelist in a cellstr, or a string for one gene
%		if a string, return a string (syslist) and a num (sysids)
% flag = 2: names from a file

if exist('flag', 'var') && flag == 2
	if exist(namelist, 'file')
		[namelist] = textread(namelist, '%s');
	else
		error(sprintf('Not exist file %s', namelist));
	end
end

strcat(Deconv.DECONV_DATASET, 'map2sys2.txt')

[names, sysnames] = textread(strcat(Deconv.DECONV_DATASET, 'map2sys2.txt'), '%s\t%s');
[stand_sys2pos] = textread(strcat(Deconv.DECONV_DATASET, 'gene.lst'), '%s');

syslist = {};
sysids = {};

if iscellstr(namelist)
	for idx=1:length(namelist)
		name = upper(namelist{idx});
		pos = strmatch(name, names, 'exact');

		n = length(pos);
		if n == 0;
			error(sprintf('Unknown identifier %s', name));
		elseif n > 1;
			error(sprintf('Ambiguous identifier %s', name));
		else
			sysid = pos(1);
			syslist{idx} = sysnames{sysid};

			std_pos = strmatch(sysnames{sysid}, stand_sys2pos, 'exact');
			sysids{idx} = std_pos;
		end
	end
else
	name = upper(namelist);

	pos = strmatch(name, names, 'exact');

	n = length(pos);
	if n == 0;
		error(sprintf('Unknown identifier %s', name));
	elseif n > 1;
		error(sprintf('Ambiguous identifier %s', name));
	else
		sysid = pos(1);
		syslist = sysnames{sysid};

		std_pos = strmatch(sysnames{sysid}, stand_sys2pos, 'exact');
		sysids = std_pos;
	end
end

return;
