import os
import pickle
import time
import gurobipy as gp

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from outputs.results import *

from src.pareto.epsilon import make_pareto_frontier, FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE
from src.huge import *
from src.compact.doctor_available import make_doctor_available_model  
from src.utils.data_gen import get_data  

import os
import pickle
import time

def epsilon_runs(model_type, problem_size, seeds, dense=True):
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
        DOCTOR_AVAILABLE: "doctor_available"
    }

    I, J, K, T = problem_size["patients"], problem_size["doctors"], problem_size["diseases"], problem_size["time periods"]

    for seed in seeds:
        filename = f"{path}/pareto_{model_names[model_type]}_seed{seed}_I{I}_J{J}_K{K}_T{T}.pkl"

        if os.path.exists(filename):
            # Load existing result
            with open(filename, "rb") as f:
                data = pickle.load(f)
            run_time = data.get("total_runtime", 0)
            print(f"[INFO] Seed {seed} already computed, loaded runtime {run_time:.2f}s")
        else:
            print(f"\n=== Running seed {seed} ===")
            data = all_data[f"seed_{seed}"]

            # Build model, variables, objectives
            m, Y, objectives, setup_time = make_doctor_available_model(data)
            m.setParam("OutputFlag", 0)

            # Compute Pareto frontier
            # dense is false
            make_pareto_frontier(data, m, Y, objectives, model_type, dense=False)

    print("[INFO] Finished computing Pareto frontiers for all seeds.")

