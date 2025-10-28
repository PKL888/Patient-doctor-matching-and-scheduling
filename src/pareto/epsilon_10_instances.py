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


def epsilon_runs(model_type, problem_size, seeds):
    # Seeds to run
    all_pareto_results = {}

    run_times = [0] * len(seeds)

    # Generate/load data
    all_data = get_data(problem_size, seeds)
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")

        # Generate/load data
        data = all_data[f"seed_{seed}"]

        start_time = time.time()

        # Build model, variables, objectives
        m, Y, objectives, setup_time = make_doctor_available_model(data)
        m.setParam("OutputFlag", 0)

        make_pareto_frontier(data, m, Y, objectives, model_type, dense=True)

        end_time = time.time()

        run_times.append(end_time - start_time)

    print("[INFO] Finished computing Pareto frontiers for all seeds.")

    return run_times