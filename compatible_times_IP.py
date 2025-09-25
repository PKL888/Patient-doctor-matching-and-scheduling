import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import random
import pickle

with open("data_seed10_I100_J10_K4_T20.pkl", "rb") as f:
    data = pickle.load(f)



# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = [t for t in range(problem_size["time periods"])]

m = gp.Model("Compatible times")

# Variables
Y = {(i,j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
    }

# Constraints
# DoctorsStartOneTreatAtATime = \
# {(j,t):
#  m.addConstr(gp.quicksum(Y.get((i,j,t),0) for k in K for i in I_k[k]) <= 1)
#  for j in J for t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION]]}

DoctorsAreNotOverbooked = \
{(j,t):
 m.addConstr(gp.quicksum(Y[i,j,tt] for k in K if j in J_k[k] for i in I_k[k] 
                         for tt in T[max(0, t - treat[j][k] + 1):t+1] if tt in compatible_times[i, j]) <= 1)
 for j in J for t in T}

PatientsAreAssignedOnlyOnce = \
{i:
 m.addConstr(gp.quicksum(Y[i,j,t] for j in J_k[k] for t in compatible_times[i,j]) <=1)
for k in K for i in I_k[k]}

# compatible_times takes care of the feasible time constraint

# FeasibleTime = \
# {(j,k,t):
#  m.addConstr(treat[j][k] * Y[i,j,t] <= sum(doctor_times[j][tt] * patient_times[i][tt] for tt in range(t, min(t + treat[j][k], len(T)))))
#  for k in K for i in I_k[k] for j in J for t in T}

# compatible_times also takes care of the doctor qualification to treat different diseases

# DoctorsQualified = \
# {(i,j,k,t):
# m.addConstr(Y[i,j,t] <= qualified[j][k])
# for k in K for i in I_k[k] for j in J for t in T}

# m.setParam("OutputFlag", 0)

# Objective 1: Max. number of matches
# print("Objective 1: Max. number of matches")

# m.setObjective(gp.quicksum(Y[i,j,t] for i in I for j in J for t in T), gp.GRB.MAXIMIZE)
#optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)

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
optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)

# Objective 3: Max. doctor satisfaction
# print("Objective 3: Max. doctor satisfaction")

# doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
# doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

# m.setObjective(gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J for t in T), gp.GRB.MAXIMIZE)
#optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times)