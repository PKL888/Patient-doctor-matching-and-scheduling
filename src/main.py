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
from utils.data_instance import DataInstance
from huge.cg_fragments_formulation import make_huge_frag_model

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
SUBSET_COLUMN_GEN = 8
FRAGMENT_COLUMN_GEN = 9

if __name__ == '__main__':
    model = int(input(f"Please enter Model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available, {SUBSET_COLUMN_GEN}: schedule_column_gen, {FRAGMENT_COLUMN_GEN}: fragment_column_gen]:    "))
    obj = int(input(f"Please enter which objective to maximise: [{NUM_APPOINTMENTS}: number_of_appointments, {PAT_SAT}: patient_satisfaction, {DOC_SAT}: doctor_satisfaction]:  "))

   

    all_data = get_data(problem_size, seeds)
    for seed in seeds:
        data = all_data[f"seed_{seed}"]
        d = DataInstance(data)

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
        if (model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]):
            # for optimise and print
            Z = None
            S = None
            F = None
            if (model == FEASIBILITY):
                m, Y, [objective_0, objective_1, objective_2], _ = make_feasibility_model(data)
            elif (model == COMPATIBLE_TIMES):
                m, Y, [objective_0, objective_1, objective_2], _ = make_compatible_times_model(data)
            elif (model == DOCTOR_AVAILABLE):
                m, Y, [objective_0, objective_1, objective_2], _ = make_doctor_available_model(data)
            else:
                # never gets here??
                print(f"model {model} not valid")
        elif model == SUBSET_COLUMN_GEN:
            use_multi = input("Use multiprocessing? (y/n)")[0] == 'y'
            print("Using multiprocessing: ", use_multi)
            Y = None
            F = None
            # actually meant to be the wrong way round, not a bug
            m, Z, [objective_1, objective_0, objective_2], _, S = make_huge_model(d, d.seed, len(d.I), len(d.J), len(d.T), len(d.K), use_multi)
        elif model == FRAGMENT_COLUMN_GEN:
            Y = None
            S = None
            # actually meant to be the wrong way round, not a bug
            max_frag_length = int(input("Maximum fragment length (at least 2, 5 and up is very slow):    "))
            m, Z, [objective_0, objective_1, objective_2], _, F = make_huge_frag_model(d, max_frag_length)

        
            
            

        # choose model(s) --> run model(s)
        if (obj not in [NUM_APPOINTMENTS, PAT_SAT, DOC_SAT]):
            break
        else:
            # print("HERE")
            if (obj == NUM_APPOINTMENTS):
                print("set objective num appointments", NUM_APPOINTMENTS)
                # print(objective_0)
                m.setObjective(objective_0, gp.GRB.MAXIMIZE)
            elif (obj == PAT_SAT):
                print("set objective pat sat", PAT_SAT)
                # print(objective_1)
                m.setObjective(objective_1, gp.GRB.MAXIMIZE)
            elif (obj ==DOC_SAT):
                print("set objective doc_sat", DOC_SAT)
                # print(objective_2)
                m.setObjective(objective_2, gp.GRB.MAXIMIZE)
            else:
                print("Not valid objective")
        model_type = 0
        # if model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]:
        #     model_type = 0
        if model == SUBSET_COLUMN_GEN:
            model_type = 1
        if model == FRAGMENT_COLUMN_GEN:
            model_type = 2

        # (check whether a data or output file already exists)
        m.setParam("OutputFlag", 1)
        optimise_and_print_schedule(model_type, m, d.M1, Y, Z, S, d.I, d.J, d.K, d.T, d.I_k, d.treat, d.allocate_rank, d.qualified, d.doctor_rank, d.patient_available, d.patient_time_prefs, d.doctor_times, d.patient_diseases)
        

        # create summary tables or comparison plots
