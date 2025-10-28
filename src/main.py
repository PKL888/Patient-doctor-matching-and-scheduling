import gurobipy as gp
import os
import subprocess

from compact.compatible_times import make_compatible_times_model 
from compact.doctor_available import make_doctor_available_model 
from compact.feasibility import make_feasibility_model 
from huge.cg_fragments_formulation import make_huge_frag_model
from huge.cg_huge import make_huge_model
from pareto.epsilon import make_pareto_frontier
from pareto.frontier import *
from utils.data_gen import get_data
from utils.data_instance import DataInstance
from utils.logging_results import optimise_and_print_schedule

# specify problem size, number of seeds --> generate data
seeds = [19]



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
        max_frag_length = int(input("Maximum fragment length (at least 2, 5 and up is very slow):    "))
        m, W, [objective_0, objective_1, objective_2], time, F = make_huge_frag_model(d, max_frag_length)
    
    else:
        print("not a valid model????")
    
    print(f"[INFO] Generating the model has taken {round(time, 4)} seconds")
    
    return m, [objective_0, objective_1, objective_2], Y, Z, W, S, F

def user_model():
    model = int(input(f"Please enter model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available, {SUBSET_COLUMN_GEN}: schedule_column_gen, {FRAGMENT_COLUMN_GEN}: fragment_column_gen]:    "))
    while model not in [FEASIBILITY,COMPATIBLE_TIMES,DOCTOR_AVAILABLE,SUBSET_COLUMN_GEN,FRAGMENT_COLUMN_GEN]:
        print(f"{model} is not a valid model.")
        model = int(input(f"Please enter model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available, {SUBSET_COLUMN_GEN}: schedule_column_gen, {FRAGMENT_COLUMN_GEN}: fragment_column_gen]:    "))
    return model

def user_objective():
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
        print("set objective as num appointments", NUM_APPOINTMENTS)
        m.setObjective(objective_0, gp.GRB.MAXIMIZE)
    elif (obj == PAT_SAT):
        print("set objective as pat sat", PAT_SAT)
        m.setObjective(objective_1, gp.GRB.MAXIMIZE)
    elif (obj == DOC_SAT):
        print("set objective as doc_sat", DOC_SAT)
        m.setObjective(objective_2, gp.GRB.MAXIMIZE)

if __name__ == '__main__':
    model = user_model()
    obj = user_objective()

    problem_size = {
        "patients": 10,
        "doctors":  4,
        "diseases": 2,
        "time periods": 10
    }

    all_data = get_data(problem_size, seeds)
    for seed in seeds:
        data = all_data[f"seed_{seed}"]
        d = DataInstance(data)

        m, [objective_0, objective_1, objective_2], Y, Z, W, S, F = make_model(model, data, d)

        if obj == PARETO:
            m.setParam("OutputFlag", 0)
            make_pareto_frontier(data, m, Y, Z, W, S, F, d.I, d.J, d.K, d.T, [objective_0, objective_1, objective_2], model, d, dense=False)
            
        else:
            set_objective(m, obj)

            model_type = 0
            # if model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]:
            #     model_type = 0
            if model == SUBSET_COLUMN_GEN:
                model_type = 1
            if model == FRAGMENT_COLUMN_GEN:
                model_type = 2

            # (check whether a data or output file already exists)
            m.setParam("OutputFlag", 1)
            optimise_and_print_schedule(model_type, model, m, Y, Z, W, S, F, data, d, show_plot=True)




