
from src.config import Config, read_yl_vst_data_rep
from src.create_models import ModelCreation


def create_dynamic_alpha_config(posteriors_filepath, alpha, replicate, name):
	"""
	This function allows to load a config from a posteriors filepath dynamically without
	the need to save and load a model file.

	This will be useful as we try different alpha parameters. Without needing to make
	hundreds of alpha model files.
	"""
	
	# Create a model config without saving to disk
	model_creator = ModelCreation(posteriors_filepath, output_model_path=None)
	model_creator.alpha = alpha
	model_creator.create_model(save=False)

	# Load the config as lines of string
	config_lines = model_creator.model_cfg.split('\n')

	# Load the data
	wt_data = read_yl_vst_data_rep(replicate)

	# model file
	config = Config(wt1=wt_data, model_wt1_lines=config_lines, name=name)
	config.replicate = replicate

	return config
