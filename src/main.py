# import all functions
import gurobipy as gp
import os
import subprocess
from utils.data_gen import get_data
from compact.compatible_times import make_compatible_times_model 
from compact.doctor_available import make_doctor_available_model 
from compact.feasibility import make_feasibility_model 
from utils.logging_results import optimise_and_print_schedule
from huge.cg_huge import make_huge_model

# specify problem size, number of seeds --> generate data
seeds = [1]

problem_size = {
    "patients": 20,
    "doctors":  4,
    "diseases": 1,
    "time periods": 10
}

# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
# models to pick from: 
#  feasibility, compatible times, doctor availability (Hamish)
# huge formulations:
#  "smart" column gen, fragments (Tyler)😊

# user specifies if they want a:
# - epsilon OR (Peleg)

# Change output stuff to work with any model (can be specified in this file) (Hamish)
#  From main we need to be able to specify outputs from running the files:
# - call performance profiles
# - gurobi output 
# - get a schedule
# - get a plot

# - comparison tables - probably in a different file

FEASIBILITY = 1
COMPATIBLE_TIMES = 2
DOCTOR_AVAILABLE = 3

NUM_APPOINTMENTS = 1
PAT_SAT = 2
DOC_SAT = 3

if __name__ == '__main__':
    model = int(input(f"Please enter Model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available]:    "))
    obj = int(input(f"Please enter which objective to maximise: [{NUM_APPOINTMENTS}: number_of_appointments, {PAT_SAT}: patient_satisfaction, {DOC_SAT}: doctor_satisfaction]:  "))

   

    all_data = get_data(problem_size, seeds)
    for seed in seeds:
        data = all_data[f"seed_{seed}"]
        globals().update(data)


        # # choose model(s) --> run model(s)
        # COMPATIBLE_TIMES = 1
        
        # SUBSET_HUGE = 11

        # model = SUBSET_HUGE

        # m, Z, [objective_0, objective_1, objective_2], _ = make_huge_model(seed, len(I), len(J), len(T), len(K), True)
        # m.setObjective(objective_1, gp.GRB.MAXIMIZE)

        # # (check whether a data or output file already exists)
        # m.setParam("OutputFlag", 1)
        # m.optimize()
        # # optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)
        


        # choose model(s) --> run model(s)
        if (model not in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]):
            # for optimise and print
            Y = None
            break
        else:
            # for optimise and print
            Z = None
            S = None
            if (model == FEASIBILITY):
                m, Y, [objective_0, objective_1, objective_2], _ = make_feasibility_model(data)
            elif (model == COMPATIBLE_TIMES):
                m, Y, [objective_0, objective_1, objective_2], _ = make_compatible_times_model(data)
            elif (model == DOCTOR_AVAILABLE):
                m, Y, [objective_0, objective_1, objective_2], _ = make_doctor_available_model(data)
            else:
                print(f"model {model} not valid")

        # choose model(s) --> run model(s)
        if (obj not in [NUM_APPOINTMENTS, PAT_SAT, DOC_SAT]):
            break
        else:
            if (obj == NUM_APPOINTMENTS):
                m.setObjective(objective_0, gp.GRB.MAXIMIZE)
            elif (obj == PAT_SAT):
                m.setObjective(objective_1, gp.GRB.MAXIMIZE)
            elif (obj ==DOC_SAT):
                m.setObjective(objective_2, gp.GRB.MAXIMIZE)
            else:
                print("Not valid objective")
        
        model_type = 0

        # (check whether a data or output file already exists)
        m.setParam("OutputFlag", 1)
        optimise_and_print_schedule(model_type, m, M1, Y, Z, S, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times, patient_diseases)
        

        # create summary tables or comparison plots
