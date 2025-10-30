import os
import pickle
import time
import gurobipy as gp

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from outputs.results import *

from src.pareto.epsilon import make_pareto_frontier, FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE, FRAGMENT_COLUMN_GEN, SUBSET_COLUMN_GEN
from src.huge import *
from compact.compatible_times import make_compatible_times_model 
from compact.doctor_available import make_doctor_available_model 
from compact.feasibility import make_feasibility_model 
from huge.cg_fragments_formulation import make_huge_frag_model
from huge.cg_huge import make_huge_model
from src.utils.data_gen import get_data  
from utils.data_instance import DataInstance
# from main import make_model

import os
import pickle
import time

def make_model(model:int, data: dict, d: DataInstance, frag_length = 0):
    Y = None
    Z = None
    S = None
    F = None
    W = None

    if model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]:
        if model == FEASIBILITY:
            m, Y, [objective_0, objective_1, objective_2], time = make_feasibility_model(d, data)
        elif model == COMPATIBLE_TIMES:
            m, Y, [objective_0, objective_1, objective_2], time = make_compatible_times_model(data)
        elif model == DOCTOR_AVAILABLE:
            m, Y, [objective_0, objective_1, objective_2], time = make_doctor_available_model(data)
    
    elif model == SUBSET_COLUMN_GEN:
        m, Z, [objective_0, objective_1, objective_2], time, S = make_huge_model(d, d.seed, len(d.I), len(d.J), len(d.K), len(d.T))
    
    elif model == FRAGMENT_COLUMN_GEN:
        print(f"{frag_length}")
        if frag_length:
            max_frag_length = frag_length
        else:
            max_frag_length = int(input("Maximum fragment length (at least 2, 5 and up is very slow):    "))
        m, W, [objective_0, objective_1, objective_2], time, F = make_huge_frag_model(d, max_frag_length)
    
    else:
        print("not a valid model????")
    
    print(f"[INFO] Generating the model has taken {round(time, 4)} seconds")

    return m, [objective_0, objective_1, objective_2], Y, Z, W, S, F

def epsilon_runs(model_type, problem_size, seeds, dense=True, frag_length = 0):
    """
    Run epsilon experiments for given seeds.
    If a seed file already exists, just read its runtime instead of recomputing.
    Returns a list of runtimes.
    """

    # Preload all data
    all_data = get_data(problem_size, seeds)

    # Folder for saving/loading results
    path = "outputs/results"
    os.makedirs(path, exist_ok=True)
    model_names = {
        FEASIBILITY: "feasibility",
        COMPATIBLE_TIMES: "compatible_times",
        DOCTOR_AVAILABLE: "doctor_available",
        SUBSET_COLUMN_GEN: "subset_column_gen",
        FRAGMENT_COLUMN_GEN: "fragment_column_gen"        
    }

    print("-"*80)
    print(f"{model_type=}, {model_names[model_type]}")
    print("-"*80)

    i, j, k, t = problem_size["patients"], problem_size["doctors"], problem_size["diseases"], problem_size["time periods"]
    max_frag_length = 0    
    if model_type == FRAGMENT_COLUMN_GEN and not max_frag_length:
        max_frag_length = int(input("Maximum fragment length (at least 2, 5 and up is very slow):    "))

    for seed in seeds:
        filename = f"{path}/pareto_{model_names[model_type]}_seed{seed}_I{i}_J{j}_K{k}_T{t}.pkl"

        if os.path.exists(filename):
            # Load existing result
            with open(filename, "rb") as f:
                data = pickle.load(f)
            run_time = data.get("total_runtime", 0)
            print(f"[INFO] Seed {seed} already computed, loaded runtime {run_time:.2f}s")
        else:
            print(f"\n=== Running seed {seed}, {i=}, {j=}, {k=}, {t=} ===")
            data = all_data[f"seed_{seed}"]
            d = DataInstance(data)

            # Build model, variables, objectives
            # m, Y, objectives, setup_time = make_doctor_available_model(data)
            m, objectives, Y, Z, W, S, F = make_model(model_type, data, d, frag_length=max_frag_length)
            m.setParam("OutputFlag", 0)

            # Compute Pareto frontier
            # dense is false
            make_pareto_frontier(data, m, Y, Z, W, S, F, d.I, d.J, d.K, d.T, objectives, model_type, d, dense = False)

    print("[INFO] Finished computing Pareto frontiers for all seeds.")

