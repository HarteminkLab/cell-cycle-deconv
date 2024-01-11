
import numpy as np
import pandas as pd

def create_model_rg1_rpostg1(posteriors_filepath):
    """

    This model is an iteration on the RG1 model defined in create_models.py

    Intention is to add a new phase called RpostG1 distinct from postG1 that Mother (C) and Daughters (D) use.
    """

    params = {}
    with open(posteriors_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:-1]:
            line_spl = line.split()
            params[line_spl[0]] = float(line_spl[1])

    mu0, lambd, delta, sigma0, sigmav, halted = (params['mu0'], params['lambda'], 
                                         params['delta'], params['sigma0'], params['sigmav'],
                                         params['halted'])

    gamma1 = params['gamma1']
    start_of_S = gamma1*lambd

    params = mu0, lambd, delta, sigma0, sigmav, 0
    model_dic = {
        "RG1": [
            {"i":[mu0, start_of_S, 49]}],
        "CG1":[
            {"t":[0, start_of_S, 49]}],
        "DG1":[
            {"b":[-delta, start_of_S, 49]}],
        "RpostG1":[
            {"i":[start_of_S, lambd, 79]}],
        "postG1":[
            {"t":[start_of_S, lambd, 79]},
            {"b":[start_of_S, lambd, 79]}]
        }

    return params, model_dic


def get_sub_interval_str(model):
    subnames = ['R', 'RG1', 'CG1', 'DG1', 'RpostG1', 'postG1']
    branches = ['i', 't', 'b']

    intervals_str = ""
    for br in branches:
        intervals_str += f"# {br}\n"
        for subname in subnames:

            if subname not in model.keys(): 
                continue

            subintervals = model[subname]
            for subintervals_dict in subintervals:
                k, seq_params = list(subintervals_dict.items())[0]
                if k == br:
                    interval = np.linspace(seq_params[0], seq_params[1], seq_params[2]+1)
                    intervals_str += ' '.join([f"{x:.4f}" for x in interval]) + '\n'
    return intervals_str


def get_model_cfg_str(params, model_dic):

    mu0, lambd, delta, sigma0, sigmav, alpha = params

    # Unused parameter beta
    beta = 0

    intervals = get_sub_interval_str(model_dic)

    s = """# lengths
mu0 %f
lambda %f
delta %f
sigma0 %f
sigmav %f
alpha %f
beta %f
# description
RG1 i 0
CG1 t 0
DG1 b 0
postRG1 i 1
postG1 t 1 b 1
%s""" % (-mu0, lambd, delta, sigma0, sigmav, alpha, beta, intervals)

    return s


def create_wt1_model():
    output_model_path = 'models/yl_cell_cycle/wt1_rg1_postrg1.label'
    posteriors_filepath = 'data/cloccs_output_yl_replicate2/posteriors.txt'

    wt2_params, rep2_model = create_model_rg1_model(posteriors_filepath)
    wt2_cfg = get_model_cfg_str(wt2_params, rep2_model)

    with open(output_model_path, 'w') as f:
        f.write(wt2_cfg)


def create_wt2_model():
    output_model_path = 'models/yl_cell_cycle/wt2_rg1_postrg1.label'
    posteriors_filepath = 'data/cloccs_output_yl_replicate2/posteriors.txt'

    wt2_params, rep2_model = create_model_rg1_model(posteriors_filepath)
    wt2_cfg = get_model_cfg_str(wt2_params, rep2_model)

    with open(output_model_path, 'w') as f:
        f.write(wt2_cfg)


def main():

    

    from src.ModelFile import ModelFile

    model_file = ModelFile()
    model_file.load_model(output_model_path)
    model_file.plot_model()
    
