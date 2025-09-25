import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import random
import pickle
import json
import time

with open("data_seed10_I100_J10_K4_T20.pkl", "rb") as f:
    data = pickle.load(f)

# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

# diseases_doctor_can_treat_at_time = {(j,t): [k for k in diseases_doctor_qualified_for[j] if t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] + 1]]
#                                      for j in J for t in T}

# patients_with_disease_that_can_be_treated_at_time = \
# {(k,t):
#     [i for i in I_k[k] if t in T[patient_available[i][START]: patient_available[i][START] + patient_available[i][DURATION]]] 
# for k in K for t in T}

m = gp.Model("Doctor availability")

# start time for pre-solver
start_time = time.time()

##################################################################################
# Variables
Y = {(i,j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
    }

# 1 if doctor is available at time t, 0 otherwise
Z = {(j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for j in J for t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION]]
    }

##################################################################################
# Constraints
PatientsAreAssignedOnlyOnce = \
{(i):
 m.addConstr(gp.quicksum(Y[i,j,t] for j in J_k[k] for t in compatible_times[i,j]) <=1)
for k in K for i in I_k[k]
}


DoctorAvailableConstraint = \
{(j,t):
m.addConstr(Z[j,t] == 
            Z[j,t-1] # previous availability
            + gp.quicksum(Y[i,j,t-treat[j][k]] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if t-treat[j][k] in compatible_times[i,j]) # incoming availability
            - gp.quicksum(Y[i,j,t] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if t in compatible_times[i,j]) # outgoing
            )
for j in J for t in T[doctor_available[j][START] + 1:doctor_available[j][START] + doctor_available[j][DURATION]]}

DoctorsStartAvailable = \
{j:
m.addConstr(Z[j,doctor_available[j][START]] == 1 - gp.quicksum(Y[i,j,doctor_available[j][START]] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if doctor_available[j][START] in compatible_times[i,j]))
for j in J}

DoctorsEndAvailable = \
{j:
m.addConstr(Z[j,doctor_available[j][START] + doctor_available[j][DURATION]-1] + 
            gp.quicksum(Y[i,j,doctor_available[j][START] + doctor_available[j][DURATION]-treat[j][k]] 
                        for k in diseases_doctor_qualified_for[j] 
                        for i in I_k[k] if doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] in compatible_times[i,j]) == 1)
for j in J}

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

# Just make sure to store presolve info:
model_results["before_presolve_info"] = before_presolve_info

m.setParam("OutputFlag", 1)  # enable log
m.setParam("LogFile", "gurobi_presolve.log")

# Objective 1: Max. number of matches
print("Objective 1: Max. number of matches")

m.setObjective(gp.quicksum(Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
#optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)
model_results["max_matches"] = optimise_and_collect("Max matches", m, Y, M1, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs)

# Objective 2: Max. patient satisfaction
print("Objective 2: Max. patient satisfaction")

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
#optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)
# m.optimize()
model_results["patient_satisfaction"] = optimise_and_collect("Max patient satisfaction", m, Y, M1, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs)


# Objective 3: Max. doctor satisfaction
print("Objective 3: Max. doctor satisfaction")

doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

m.setObjective(gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
#optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)
model_results["doctor_satisfaction"] = optimise_and_collect("Max doctor satisfaction", m, Y, M1, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs)

# write model results into json file
with open("all_model_results.json", "w") as f:
    json.dump(model_results, f, indent=4)

