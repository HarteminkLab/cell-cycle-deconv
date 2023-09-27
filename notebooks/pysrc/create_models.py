
import numpy as np
import pandas as pd


def create_model_config_1_1_1(filepath):
    """ models.cfg defines how the R, CG1, DG1, postG1 interavals are created.

     Specifically, I don't believe the cfg files are different for 1.1.1.1 and 1.1.1. We use 
         1.1.1.26 where 26 is the alpha value (time for bud release, see the paper for more info)

      
            "1.1.1":{
                "R":[{"i":["-mu0","-alpha","88"]}],
                
                "CG1":[
                    {"t":["-alpha","lambda*beta","42"]},
                    {"i":["-alpha","lambda*beta","42"]}],
                
                "DG1":[{"b":["-delta-alpha","beta*lambda","42+44"]}]
                "postG1":[
                    {"t":["lambda*beta","-alpha+lambda","42"]},
                    {"i":["lambda*beta","-alpha+lambda","42"]},
                    {"b":["lambda*beta","-alpha+lambda","42"]}],
            },
    """
    params = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:-1]:
            line_spl = line.split()
            params[line_spl[0]] = float(line_spl[1])

    mu0, lambd, delta, sigma0, sigmav, halted = (params['mu0'], params['lambda'], 
                                         params['delta'], params['sigma0'], params['sigmav'], 
                                         params['halted'])

    # TODO: Investigate the importance of these values
    alpha = 0
    beta = params['gamma1']

    return (mu0, lambd, delta, sigma0, sigmav, alpha, beta), {
        "H": [{'h': [mu0-1, mu0, 1] }],
        "R":[{"i":[mu0, -alpha, 88]}],

        "CG1":[
            {"t":[-alpha, lambd*beta, 42]},
            {"i":[-alpha, lambd*beta, 42]}],

        "DG1":[{"b":[-delta-alpha, beta*lambd, 42+44]}],
        "postG1":[
            {"t":[lambd*beta, -alpha+lambd, 42]},
            {"i":[lambd*beta, -alpha+lambd, 42]},
            {"b":[lambd*beta, -alpha+lambd, 42]}],
    }


def get_sub_interval_str(model):
    subnames = ['H', 'R', 'RG1', 'CG1', 'DG1', 'postG1']
    branches = ['h', 'i', 't', 'b']

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


def get_model_cfg_str(params, model, Rname="R", CG1_intervals="i 1 t 0", PG1_intervals="i 2 t 1 b 1"):

    mu0, lambd, delta, sigma0, sigmav, alpha, beta, halted = params

    intervals = get_sub_interval_str(model)

    s = """# lengths
mu0 %f
lambda %f
delta %f
sigma0 %f
sigmav %f
alpha %f
beta %f
halted %f
# description
H h 0
%s i 0
CG1 %s
DG1 b 0
postG1 %s
%s""" % (-mu0, lambd, delta, sigma0, sigmav, alpha, beta, halted,
         Rname, CG1_intervals, PG1_intervals, intervals)

    return s


def create_model_config_1_2_1(filepath):
    """

    The RG1 model modeled after the original 1.2.1 model in the deconvolution code. With added halted cells

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
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:-1]:
            line_spl = line.split()
            params[line_spl[0]] = float(line_spl[1])

    mu0, lambd, delta, sigma0, sigmav, halted = (params['mu0'], params['lambda'], 
                                         params['delta'], params['sigma0'], params['sigmav'],
                                         params['halted'])

    # TODO: Investigate the importance of these values
    alpha = 0
    gamma1 = params['gamma1']
    sStart = beta = gamma1*lambd

    print(f"mu0: {mu0}")
    print(f"lambd: {lambd}")
    print(f"delta: {delta}")
    print()

    params = mu0, lambd, delta, sigma0, sigmav, alpha, beta, halted
    model = {
        "H": [
            {'h': [0, lambd, 1] }],
        "RG1": [
            {"i":[mu0, sStart, 32]}],
        "CG1":[
            {"t":[-alpha, sStart, 42]}],
        "DG1":[
            {"b":[-delta-alpha, sStart, 52]}],
        "postG1":[
            {"t":[sStart, -alpha+lambd, 42]},
            {"i":[sStart, -alpha+lambd, 42]},
            {"b":[sStart, -alpha+lambd, 42]}],
        }

    return params, model

