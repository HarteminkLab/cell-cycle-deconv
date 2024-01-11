
import numpy as np
import pandas as pd

#
# This class is a port of the model creation from the original matlab code. There are some changes, as we had to port to Python
# But most of the meat is here. In our case, we are only concerned with creating the RG1 model, so we will be refactoring this 
# file to simplify things.
#

def create_model_rg1_model(posteriors_filepath):
    """

    The RG1 model modeled after the original 1.2.1 model in the deconvolution code

    # ------------------
    # 1.2.1 MODEL (ALL_DIFF_G1, ibt)
    # ------------------ 
    my ($diff_g1_rg1, $diff_g1_cg1, $diff_g1_dg1, $diff_g1_postg1) = (192, 64, 192, 64);
    print OUT <<DIFF_G1;
            "1.2.1":{
                "RG1":[{"i":["-mu0","lambda*beta","$diff_g1_rg1"]}],
                "CG1":[{"t":["-alpha","lambda*beta","$diff_g1_cg1"]}],
                "DG1":[{"b":["-delta-alpha","lambda*beta","$diff_g1_dg1"]}],
                "postG1":[{"t":["lambda*beta","-alpha+lambda","$diff_g1_postg1"]},
                    {"b":["lambda*beta","-alpha+lambda","$diff_g1_postg1"]},
                    {"i":["lambda*beta","-alpha+lambda","$diff_g1_postg1"]}]
            },
    DIFF_G1
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
        "postG1":[
            {"t":[start_of_S, lambd, 79]},
            {"i":[start_of_S, lambd, 79]},
            {"b":[start_of_S, lambd, 79]}],
        }

    return params, model_dic


def get_sub_interval_str(model):
    subnames = ['R', 'RG1', 'CG1', 'DG1', 'postG1']
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


def get_model_cfg_str(params, model_dic, Rname="R", CG1_intervals="t 0", PG1_intervals="i 1 t 1 b 1"):

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
%s i 0
CG1 %s
DG1 b 0
postG1 %s
%s""" % (-mu0, lambd, delta, sigma0, sigmav, alpha, beta,
         Rname, CG1_intervals, PG1_intervals, intervals)

    return s


def create_wt1_model():
    output_model_path = 'models/yl_cell_cycle/wt1_rg1.label'
    posteriors_filepath = 'data/cloccs_output_yl_replicate1/posteriors.txt'

    wt1_params, rep2_model = create_model_rg1_model(posteriors_filepath)
    wt1_cfg = get_model_cfg_str(wt1_params, rep2_model)

    with open(output_model_path, 'w') as f:
        f.write(wt1_cfg)

    print(f"Created model and saved to file: {output_model_path}")

    return output_model_path


def create_wt2_model():
    output_model_path = 'models/yl_cell_cycle/wt2_rg1.label'
    posteriors_filepath = 'data/cloccs_output_yl_replicate2/posteriors.txt'

    wt2_params, rep2_model = create_model_rg1_model(posteriors_filepath)
    wt2_cfg = get_model_cfg_str(wt2_params, rep2_model)

    with open(output_model_path, 'w') as f:
        f.write(wt2_cfg)

    print(f"Created model and saved to file: {output_model_path}")

    return output_model_path


def main():

    create_wt1_model()
    create_wt2_model()

    # If we want to plot the resulting models, we can use this code
    # from src.ModelFile import ModelFile
    # model_file = ModelFile()
    # model_file.load_model(output_model_path)
    # model_file.plot_model()
    