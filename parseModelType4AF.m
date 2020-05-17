function [model] = parseModelType4AF(model)

modeltype = upper(model.modeltype);

all_models = { ...
	'NORMAL' ...
	'RCC' ...
};

if numel(strmatch(modeltype, all_models, 'exact'))>0
	modelprefix = strcat(modeltype, '_MODEL');
else
	error('MODEL type error');
end

model.modelprefix = modelprefix;

return;
