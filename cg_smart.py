import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import random
import pickle
import json
import time
import itertools

from typing import Dict, FrozenSet, Tuple


with open("data_seed10_I100_J10_K4_T20.pkl", "rb") as f:
    data = pickle.load(f)

# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

START = 0
DURATION = 1

"""
Finds all the patient

Returns: A set of sets of patients for this doctor, with the scores for each objective

"""
def find_all_patient_sets_for_doctor(doctor: int) -> Dict[FrozenSet[int], Tuple[int]]:
    pass 

S = dict()
for j in J:
    S[j] = find_all_patient_sets_for_doctor(j)




"""
Finds the best schedule for these patients for this doctor

If it is infeasible, the objective values are not set.

Returns: (true, paitents, (objective_value 1, val 2, val 3))
"""
def find_best_schedule(doctor: int, patients:set[int]) -> tuple[bool, dict[tuple[int, int, int], int], tuple[float]]:
    # create model
    m = gp.Model("small MIP")

    # Variables
    Y = {(i,doctor,t):
        m.addVar(vtype=gp.GRB.BINARY)
        for k in K for i in I_k[k] if i in patients for t in compatible_times[i,doctor]
        }

    # 1 if doctor is available at time t, 0 otherwise
    Z = {(doctor,t):
        m.addVar(vtype=gp.GRB.BINARY)
        for t in T[doctor_available[doctor][START]:doctor_available[doctor][START] + doctor_available[doctor][DURATION]]
        }
    
    # Constraints
    AllPatientsSeenOnce = \
    {(i):
     m.addConstr(gp.quicksum(Y[i, doctor, t] for t in compatible_times[i, doctor]) >= 1) 
     for i in I if i in patients}

    PatientsAreAssignedOnlyOnce = \
    {(i):
    m.addConstr(gp.quicksum(Y[i,doctor,t] for t in compatible_times[i,doctor]) <=1)
    for k in K for i in I_k[k] if i in patients
    }


    DoctorAvailableConstraint = \
    {(doctor,t):
    m.addConstr(Z[doctor,t] == 
                Z[doctor,t-1] # previous availability
                + gp.quicksum(Y[i,doctor,t-treat[doctor][k]] for k in diseases_doctor_qualified_for[doctor] for i in I_k[k] if i in patients if t-treat[doctor][k] in compatible_times[i,doctor]) # incoming availability
                - gp.quicksum(Y[i,doctor,t] for k in diseases_doctor_qualified_for[doctor] for i in I_k[k] if i in patients if t in compatible_times[i,doctor]) # outgoing
                )
    for t in T[doctor_available[doctor][START] + 1:doctor_available[doctor][START] + doctor_available[doctor][DURATION]]}

    DoctorsStartAvailable = m.addConstr(Z[doctor,doctor_available[doctor][START]] == 1 - gp.quicksum(Y[i,doctor,doctor_available[doctor][START]] for k in diseases_doctor_qualified_for[doctor] for i in I_k[k] if i in patients if doctor_available[doctor][START] in compatible_times[i,doctor]))

    DoctorsEndAvailable = m.addConstr(Z[doctor,doctor_available[doctor][START] + doctor_available[doctor][DURATION]-1] + 
                gp.quicksum(Y[i,doctor,doctor_available[doctor][START] + doctor_available[doctor][DURATION]-treat[doctor][k]] 
                            for k in diseases_doctor_qualified_for[doctor] 
                            for i in I_k[k] if i in patients if doctor_available[doctor][START] + doctor_available[doctor][DURATION] - treat[doctor][k] in compatible_times[i,doctor]) == 1)



    # objectives
    m.setObjective(gp.quicksum(Y[i,doctor,t] for k in K for i in I_k[k] if i in patients for t in compatible_times[i,doctor]), gp.GRB.MAXIMIZE)
    m.optimize()

    if m.status != gp.GRB.OPTIMAL:
        return False, None, None

    obj1 = m.ObjVal
    #print(obj1)

    patientDoctorScore = {i: allocate_rank[i][doctor] for i in I if i in patients}
    patientTimeScore = {i: [(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I if i in patients}

    m.setObjective(gp.quicksum(Y[i,doctor,t] * 
                            (
                                    patientDoctorScore[i]
                                    + 
                                    sum(patientTimeScore[i][t:min(t + treat[doctor][k], len(T))]) / treat[doctor][k]
                                    )
                           for k in K for i in I_k[k] if i in patients for t in compatible_times[i,doctor]), gp.GRB.MAXIMIZE)
    m.optimize()

    if m.status != gp.GRB.OPTIMAL:
        return False, None, None

    obj2 = m.ObjVal
    #print(obj2)

    doctor_num_diseases_can_treat = sum(qualified[doctor])
    doctor_disease_rank_scores = [qualified[doctor][k] * (doctor_num_diseases_can_treat - doctor_rank[doctor][k] + 1)/doctor_num_diseases_can_treat + (1 - qualified[doctor][k]) * -M1 for k in K]

    m.setObjective(gp.quicksum((doctor_disease_rank_scores[k]) * Y[i,doctor,t] for k in K for i in I_k[k] if i in patients for t in compatible_times[i,doctor]), gp.GRB.MAXIMIZE)

    m.optimize()

    if m.status != gp.GRB.OPTIMAL:
        return False, None, None

    obj3 = m.ObjVal
    #print(obj3)

    Y_values = {(i, d, t): Y[i, d, t].x for (i, d, t) in Y if Y[i, d, t].x >= 0.9}
    return (True, Y_values, (obj1, obj2, obj3))

print(find_best_schedule(1, {1,2}))
# set of schedules


# mip







m = gp.Model("Doctor availability")