import numpy as np
import os

class Config:

    def __init__(self, wt1_tp, wt2_tp, data_wt1_file, data_wt2_file, gene_mapping_file, gene_set_file, model_wt1_file, model_wt2_file):
        # deconv constants
        self.WAVELET = 1
        self.DIFF = 2
        self.WT1 = 'WT1'
        self.WT2 = 'WT2'
        self.JOINT = 'JOINT'
        self.KERNEL = 1

        self.WT1_TIMEPOINTS = wt1_tp
        self.WT2_TIMEPOINTS = wt2_tp
        
        # load datasets
        self.data_wt1 = np.loadtxt(data_wt1_file, delimiter='\t', dtype=np.float64)
        self.data_wt2 = np.loadtxt(data_wt2_file, delimiter='\t', dtype=np.float64)

        # load gene/orf mappings
        self.gene_orf_map = self.readGeneOrfMap(gene_mapping_file)
        self.orf_index_map = self.readOrfIndexMap(gene_set_file)

        self.model_intervals_1 = self.read_model_format(model_wt1_file)
        self.model_intervals_2 = self.read_model_format(model_wt2_file)

    def readGeneOrfMap(self, gene_mapping_file):
        map = {}
        with open(gene_mapping_file) as f:
            for line in f:
                (key, val) = line.strip().split('\t')
                map[key] = val
        return map

    def readOrfIndexMap(self, gene_set_file):
        map = {}
        with open(gene_set_file) as f:
            map = {orf: index for index, orf in enumerate(f.read().split('\n'))}
        return map
    
    def read_model_format(self, modelfile):
        lengths = []

        if not os.path.exists(modelfile):
            raise FileNotFoundError(f"The model file {modelfile} does not exist")

        # headers
        LENGTHS = '# lengths'
        DESCRIPTION = '# description'
        I = '# i'
        T = '# t'
        B = '# b'

        iList, tList, bList, relations = [], [], [], []
        parseFlag = -1

        with open(modelfile, 'r') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line == LENGTHS:
                    parseFlag = 1
                elif line == DESCRIPTION:
                    parseFlag = 2
                elif line == I:
                    parseFlag = 3
                elif line == T:
                    parseFlag = 4
                elif line == B:
                    parseFlag = 5
                # lengths
                elif parseFlag == 1:
                    flag, value = self.parse_lengths(line)
                    if flag == 0:
                        return
                    else:
                        lengths.append(value)
                # description
                elif parseFlag == 2:
                    relation = line.split(' ')
                    relations.append(relation)
                # interval i
                elif parseFlag == 3:
                    interval = np.array(line.split(' '), dtype=np.float64)
                    iList.append(interval)
                # interval t
                elif parseFlag == 4:
                    interval = np.array(line.split(' '), dtype=np.float64)
                    tList.append(interval)
                # interval b
                elif parseFlag == 5:
                    interval = np.array(line.split(' '), dtype=np.float64)
                    bList.append(interval)
                elif parseFlag == -1:
                    print(f"Error in line {line} ... exiting")
                    flag = 0
                    return

        i_intervals = {}
        t_intervals = {}
        b_intervals = {}

        for i, relation in enumerate(relations):
            notation = relation[0]

            for idx in range(1, len(relation)-1, 2):
                label = relation[idx]
                num = int(relation[idx+1])

                if label == 'i':
                    i_intervals[num] = (notation, i)
                elif label == 't':
                    t_intervals[num] = (notation, i)
                elif label == 'b':
                    b_intervals[num] = (notation, i)

        return lengths, relations, iList, tList, bList, i_intervals, t_intervals, b_intervals


    def parse_lengths(self, line):
        segments = line.split(' ')
        if segments[0] == 'mu0':
            value = float(segments[1])
            flag = 1
        elif segments[0] == 'lambda':
            value = float(segments[1])
            flag = 1
        elif segments[0] == 'delta':
            value = float(segments[1])
            flag = 1
        elif segments[0] == 'sigma0':
            value = float(segments[1])
            flag = 1
        elif segments[0] == 'sigmav':
            value = float(segments[1])
            flag = 1
        elif segments[0] == 'alpha':
            value = float(segments[1])
            flag = 1
        elif segments[0] == 'beta':
            value = float(segments[1])
            flag = 1
        else:
            print(f"Error in line {line} ... exiting")
            flag = 0
            value = 0
        return flag, value