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

# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
# models to pick from: 
#  feasibility, compatible times, doctor availability (Hamish)
# huge formulations:
#  "smart" column gen, fragments (Tyler)

# user specifies if they want a:
# - epsilon OR (Peleg)

# Change output stuff to work with any model (can be specified in this file) (Hamish)
#  From main we need to be able to specify outputs from running the files:
# - call performance profiles
# - gurobi output 
# - get a schedule
# - get a plot

# - comparison tables - probably in a different file




all_data = get_data(problem_size, seeds)
for seed in seeds:
    data = all_data[f"seed_{seed}"]
    globals().update(data)


    # choose model(s) --> run model(s)
    COMPATIBLE_TIMES = 1

    model = COMPATIBLE_TIMES

    m, Y, [objective_0, objective_1, objective_2], _ = make_compatible_times_model(data)
    m.setObjective(objective_1, gp.GRB.MAXIMIZE)

    # (check whether a data or output file already exists)
    m.setParam("OutputFlag", 1)
    optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)
    

# create summary tables or comparison plots
