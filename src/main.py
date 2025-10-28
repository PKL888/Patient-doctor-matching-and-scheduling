import gurobipy as gp
import pickle
import os
import subprocess
import numpy as np

from compact.compatible_times import make_compatible_times_model 
from compact.doctor_available import make_doctor_available_model 
from compact.feasibility import make_feasibility_model 
from huge.cg_fragments_formulation import make_huge_frag_model
from huge.cg_huge import make_huge_model
from pareto.epsilon import make_pareto_frontier
from pareto.frontier import *
from pareto.epsilon_10_instances import epsilon_runs
from utils.data_gen import get_data
from utils.data_instance import DataInstance
from utils.logging_results import optimise_and_print_schedule
from utils.graph_1000_seed_model_results import plot_model_files
from utils.print_model_results import summarize_results
from utils.print_model_results import summarize_pareto_slack_results
# import output result data
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from outputs.results import *
from src.huge import *

# specify problem size, number of seeds --> generate data
seeds = [1]

problem_size = {
    "patients": 100,
    "doctors":  10,
    "diseases": 4,
    "time periods": 20
}

FEASIBILITY = 1
COMPATIBLE_TIMES = 2
DOCTOR_AVAILABLE = 3
SUBSET_COLUMN_GEN = 8
FRAGMENT_COLUMN_GEN = 9

NUM_APPOINTMENTS = 1
PAT_SAT = 2
DOC_SAT = 3
PARETO = 5



def make_model(model:int, data: dict, d: DataInstance):
    Y = None
    Z = None
    S = None
    F = None
    W = None
    if (model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]):
        # for optimise and print

        if (model == FEASIBILITY):
            m, Y, [objective_0, objective_1, objective_2], time = make_feasibility_model(data)
        elif (model == COMPATIBLE_TIMES):
            m, Y, [objective_0, objective_1, objective_2], time = make_compatible_times_model(data)
        elif (model == DOCTOR_AVAILABLE):
            m, Y, [objective_0, objective_1, objective_2], time = make_doctor_available_model(data)
        else:
            # never gets here??
            print(f"model {model} not valid")
    elif model == SUBSET_COLUMN_GEN:
        # actually meant to be the wrong way round, not a bug
        m, Z, [objective_1, objective_0, objective_2], time, S = make_huge_model(d, d.seed, len(d.I), len(d.J), len(d.T), len(d.K))
    else:
        if model != model == FRAGMENT_COLUMN_GEN:
            print("not a valid model????")

        # actually meant to be the wrong way round, not a bug
        max_frag_length = int(input("Maximum fragment length (at least 2, 5 and up is very slow):    "))
        m, W, [objective_0, objective_1, objective_2], time, F = make_huge_frag_model(d, max_frag_length)
    print(f"[INFO] Generating the model has taken {round(time, 4)} seconds")
    return m, [objective_0, objective_1, objective_2], Y, Z, W, S, F

def get_model():
    model = int(input(f"Please enter Model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available, {SUBSET_COLUMN_GEN}: schedule_column_gen, {FRAGMENT_COLUMN_GEN}: fragment_column_gen]:    "))
    while model not in [FEASIBILITY,COMPATIBLE_TIMES,DOCTOR_AVAILABLE,SUBSET_COLUMN_GEN,FRAGMENT_COLUMN_GEN]:
        print(f"{model} is not a valid model.")
        model = int(input(f"Please enter Model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available, {SUBSET_COLUMN_GEN}: schedule_column_gen, {FRAGMENT_COLUMN_GEN}: fragment_column_gen]:    "))
    return model

def get_objective():
    obj = int(input(f"Please enter objective: [{NUM_APPOINTMENTS}: number of appointments, {PAT_SAT}: patient satisfaction, {DOC_SAT}: doctor satisfaction, {PARETO}: pareto frontier]:    "))
    while obj not in [NUM_APPOINTMENTS,PAT_SAT,DOC_SAT,PARETO]:
        print(f"{obj} is not a valid objective.")
        obj = int(input(f"Please enter objective: [{NUM_APPOINTMENTS}: number of appointments, {PAT_SAT}: patient satisfaction, {DOC_SAT}: doctor satisfaction, {PARETO}: pareto frontier]:    "))
    return obj

def set_objective(m: gp.Model, obj: int):
    if (obj not in [NUM_APPOINTMENTS, PAT_SAT, DOC_SAT]):
        print(f"{obj} is not a valid objective")
        return
    
    if (obj == NUM_APPOINTMENTS):
        print("set objective num appointments", NUM_APPOINTMENTS)
        m.setObjective(objective_0, gp.GRB.MAXIMIZE)
    elif (obj == PAT_SAT):
        print("set objective pat sat", PAT_SAT)
        m.setObjective(objective_1, gp.GRB.MAXIMIZE)
    elif (obj == DOC_SAT):
        print("set objective doc_sat", DOC_SAT)
        m.setObjective(objective_2, gp.GRB.MAXIMIZE)
    else:
        print("Not valid objective, should not have reached here")

if __name__ == '__main__':
    model = get_model()
    print("Model:", model)
    obj = get_objective()
   

    all_data = get_data(problem_size, seeds)
    for seed in seeds:
        data = all_data[f"seed_{seed}"]
        d = DataInstance(data)

        m, [objective_0, objective_1, objective_2], Y, Z, W, S, F \
        = make_model(model, data, d)

        if obj == PARETO:
            m.setParam("OutputFlag", 0)
            make_pareto_frontier(data, m, Y, [objective_0, objective_1, objective_2], model, dense=False)
            
        
        else:
            set_objective(m, obj)

            model_type = -1
            if model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]:
                model_type = 0
            if model == SUBSET_COLUMN_GEN:
                model_type = 1
            if model == FRAGMENT_COLUMN_GEN:
                model_type = 2

            # (check whether a data or output file already exists)
            m.setParam("OutputFlag", 1)
            optimise_and_print_schedule(model_type, seed, model, m, d.M1, Y, Z, S, d.I, d.J, d.K, d.T, d.I_k, d.treat, d.allocate_rank, d.qualified, d.doctor_rank, d.patient_available, d.patient_time_prefs, d.doctor_times, d.patient_diseases)
        
        
    # plot basic models
    plot = int(input("Do you want to print the compact model performance profiles or column generation: 1: Compact Models, 2: Column Generation, 3: None       "))
    if (plot == 1):
        model_files = {
            "Feasibility": "outputs/results/F_all_1000_seeds_model_results.pkl",
            "Compatible times": "outputs/results/CT_all_1000_seeds_model_results.pkl",
            "Doctor available": "outputs/results/DA_all_1000_seeds_model_results.pkl",
        }

        plot_model_files(model_files, 1)
    elif (plot == 2):
        model_files = {
            "Singular:": "src/huge/cg_schedules_timed_all_100_seeds_model_results.pkl",
            "Multiproccessing: ": "src/huge/cg_schedules_timed_multiproccessing_all_100_seed_model_results.pkl"
        }
        
        plot_model_files(model_files, 2)

    pareto_table = int(input("Do you want to print Epsilon Model table output fo 10 instances: 1: Yes, 2: No     "))
    if (pareto_table == 1):

        epsilon_problem_size = {
            "patients": 50,
            "doctors":  5,
            "diseases": 3,
            "time periods": 20
        }

        # Seeds 1 through 10
        seeds = range(1, 11)
        
        run_times = epsilon_runs(DOCTOR_AVAILABLE, epsilon_problem_size, seeds)

        # Call function to summarize results
        summarize_pareto_slack_results(seeds, epsilon_problem_size["patients"], epsilon_problem_size["doctors"],
                                       epsilon_problem_size["diseases"], epsilon_problem_size["time periods"])

        # Convert to numpy array for convenience
        run_times = np.array(run_times, dtype=float)

        # Compute statistics
        mean_time = np.mean(run_times)
        min_time = np.min(run_times)
        max_time = np.max(run_times)
        std_time = np.std(run_times)

        # Print nicely
        print(f"Run times (s): mean = {mean_time:.2f}, std = {std_time:.2f}, min = {min_time:.2f}, max = {max_time:.2f}")
                
       
            



