# import all functions
import gurobipy as gp
import os
import subprocess
from utils.data_gen import get_data
from compact.compatible_times import make_compatible_times_model 
from utils.logging_results import optimise_and_print_schedule

# specify problem size, number of seeds --> generate data
seeds = [1]

problem_size = {
    "patients": 100,
    "doctors":  10,
    "diseases": 4,
    "time periods": 20
}

all_data = get_data(problem_size, seeds)
# print("-"*100)
for seed in seeds:
    data = all_data[f"seed_{seed}"]
    # print(data.keys())
    globals().update(data)


    # print("-"*100)
    
    # choose model(s) --> run model(s)
    COMPATIBLE_TIMES = 1

    model = COMPATIBLE_TIMES

    m, Y, [objective_0, objective_1, objective_2], _ = make_compatible_times_model(data)
    m.setObjective(objective_1, gp.GRB.MAXIMIZE)

    # (check whether a data or output file already exists)
    m.setParam("OutputFlag", 1)
    optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)
    

# create summary tables or comparison plots