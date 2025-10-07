import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import random
import pickle
import time

with open("all_data_1000_seeds_I100_J10_K4_T20.pkl", "rb") as f:
    all_data = pickle.load(f)

all_model_results = {}
objectives = ["max_matches", "patient_satisfaction", "doctor_satisfaction"]
all_model_results = {obj: {} for obj in objectives}

for obj in objectives:
        
    for seed, data in all_data.items():

        # put everything in the global namespace
        globals().update(data)

        I = range(problem_size["patients"])
        J = range(problem_size["doctors"])
        K = range(problem_size["diseases"])
        T = range(problem_size["time periods"])

        m = gp.Model("Compatible times")

        start_time = time.time()

        # Variables
        Y = {(i,j,t):
            m.addVar(vtype=gp.GRB.BINARY)
            for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
            }
        
        DoctorsAreNotOverbooked = \
        {(j,t):
        m.addConstr(gp.quicksum(Y[i,j,tt] for k in K if j in J_k[k] for i in I_k[k] 
                                for tt in T[max(0, t - treat[j][k] + 1):t+1] if tt in compatible_times[i, j]) <= 1)
        for j in J for t in T}

        PatientsAreAssignedOnlyOnce = \
        {i:
        m.addConstr(gp.quicksum(Y[i,j,t] for j in J_k[k] for t in compatible_times[i,j]) <=1)
        for k in K for i in I_k[k]}

        #################################################################
        # printing and optimising

        m.setParam("OutputFlag", 0)

        model_results = {}

        m.update()

        # Record before presolve info
        setup_time = time.time() - start_time
        before_presolve_info = {
            "num_variables": m.NumVars,
            "num_constraints": m.NumConstrs,
            "num_nonzeros": m.NumNZs,
            "setup_time_seconds": setup_time
        }

        m.setParam("OutputFlag", 0)  # enable log
        m.setParam("LogFile", "gurobi_presolve.log")

        # Set the objective depending on `obj`
        if obj == "max_matches":
            # Objective 1: Max. number of matches
            #print("Objective 1: Max. number of matches")
            m.setObjective(gp.quicksum(Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)

        # Set the objective depending on `obj`
        elif obj == "patient_satisfaction":
            # Objective 2: Max. patient satisfaction
            #print("Objective 2: Max. patient satisfaction")

            numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
            patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
            patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]

            m.setObjective(gp.quicksum(Y[i,j,t] * 
                                    (
                                            patientDoctorScore[i][j] 
                                            + 
                                            sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]
                                        )
                                    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)


        # Set the objective depending on `obj`
        elif obj == "doctor_satisfaction":

            doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
            doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

            m.setObjective(gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
            #optimise_and_print_schedule()

        # Solve and collect results
        model_results = optimise_and_collect(obj, m, Y, M1, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs)

        # Store results
        all_model_results[obj][seed] = {
            "before_presolve_info": before_presolve_info,
            "model_results": model_results
        }

    # write all model results into json file
    with open("CT_all_1000_seeds_model_results.pkl", "wb") as f:
        pickle.dump(all_model_results, f)

