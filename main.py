from config import Config
from model import Model

# YL_CELL_CYCLE
YL_WT1_TP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140]
YL_WT2_TP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140]
# dataset files
YL_DATA_WT1_FILE = 'datasets/yl_cell_cycle/replicate2_gene_expression.txt'
YL_DATA_WT2_FILE = 'datasets/yl_cell_cycle/replicate2_gene_expression.txt'
YL_GENE_MAPPING_FILE = 'datasets/yl_cell_cycle/gene_to_orf_name_mapping.txt'
YL_GENE_SET_FILE = 'datasets/yl_cell_cycle/genes.lst'
# model files
YL_MODEL_WT1_FILE = 'models/yl_cell_cycle/wt2_rg1.label'
YL_MODEL_WT2_FILE = 'models/yl_cell_cycle/wt2_rg1.label'

# ORIGINAL_BUDFLOW
OG_WT1_TP = [x for x in range(30, 255, 16)]
OG_WT2_TP = [x for x in range(38, 263, 16)]
# dataset files
DATASET_PATH = 'datasets/original_budflow/'
OG_DATA_WT1_FILE = 'datasets/original_budflow/replicate1_gene_expression.txt'
OG_DATA_WT2_FILE = 'datasets/original_budflow/replicate2_gene_expression.txt'
OG_GENE_MAPPING_FILE = 'datasets/original_budflow/gene_to_orf_name_mapping.txt'
OG_GENE_SET_FILE = 'datasets/original_budflow/genes.lst'
# model files
OG_MODEL_WT1_FILE = 'models/original_budflow/wt1_budflow/1.1.1.26.label'
OG_MODEL_WT2_FILE = 'models/original_budflow/wt2_budflow/1.1.1.27.label'

if __name__ == "__main__":
    # yl_config = Config(YL_WT1_TP, YL_WT2_TP, YL_DATA_WT1_FILE, YL_DATA_WT2_FILE, 
    #                    YL_GENE_MAPPING_FILE, YL_GENE_SET_FILE, YL_MODEL_WT1_FILE, YL_MODEL_WT2_FILE)
    # yl_model = Model(yl_config, 'CLN2', 0.004)

    og_config = Config(OG_WT1_TP, OG_WT2_TP, OG_DATA_WT1_FILE, OG_DATA_WT2_FILE, 
                       OG_GENE_MAPPING_FILE, OG_GENE_SET_FILE, OG_MODEL_WT1_FILE, OG_MODEL_WT2_FILE)
    og_model = Model(og_config, 'CLN2', 0.00429)

    # og_model.deconvolve()