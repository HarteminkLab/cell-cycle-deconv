import unittest

import numpy as np
import matlab_values
from config import Config
from model import Model

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

# class TestConfig(unittest.TestCase):

#     def test_gene_orf_map(self):
#         pass

#     def test_orf_index_map(self):
#         pass

#     def test_read_model_data(self):
#         pass

class TestModel(unittest.TestCase):

    def setUp(self):
        config = Config(OG_WT1_TP, OG_WT2_TP, OG_DATA_WT1_FILE, OG_DATA_WT2_FILE, 
                       OG_GENE_MAPPING_FILE, OG_GENE_SET_FILE, OG_MODEL_WT1_FILE, OG_MODEL_WT2_FILE)
        self.model = Model(config, 'CLN2', 0.00429)

    def test_calc_H(self):
        # self.assertTrue(np.allclose(self.model.H, matlab_values.H))
        pass

    # def test_deconvolve(self):
    #     pass

if __name__ == '__main__':
    unittest.main()