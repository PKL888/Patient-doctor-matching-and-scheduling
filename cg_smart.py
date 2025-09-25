import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import random
import pickle
import json
import time
import itertools

from typing import Dict, FrozenSet, Tuple, Optional

with open("data_seed10_I20_J4_K2_T10.pkl", "rb") as f:
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
Finds the best schedule for these patients for this doctor

If it is infeasible, the objective values are not set.

Returns: (true, paitents, (objective_value 1, val 2, val 3))
"""
def find_best_schedule(doctor: int, patients:set[int]) -> tuple[bool, Optional[dict[tuple[int, int, int], int]], Optional[tuple[float, float, float]]]:
    # create model
    m = gp.Model("small MIP")
    m.setParam("OutputFlag", 0)

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



"""
Finds all the patient

Returns: A set of sets of patients for this doctor, with the scores for each objective

"""
def find_all_patient_sets_for_doctor(doctor: int) -> Dict[FrozenSet[int], Tuple[float, float, float]]:
    # print("Doctor:", j)
    patients = patients_doctor_can_treat[j]
    # print(len(patients))

    schedules_n_patients = dict()
    #                             (set with no patients, tuple of 0 objective values)
    #                         change to not be sets????
    schedules_n_patients[0] = [ ([], (0,0,0)) ]
    schedules_n_patients[1] = [([patient], find_best_schedule(doctor, {patient})[2]) for patient in patients]
    n = 2
    while True:
        # print("n:", n)
        schedules_n_patients[n] = []
        for patient_list, _ in schedules_n_patients[n - 1]:
            
            # {([1], (obj0, obj1, obj2)), ([2], (obj0, obj1, obj2))}
            # schedule is ([1], (obj0, obj1, obj2))

            last_patient = patient_list[-1]
            potential_patients = [p for p in patients if p > last_patient]
            for patient in potential_patients:
                new_patient_list = patient_list + [patient]
                                                                           # make take in a list???
                feasible, _, objective_values = find_best_schedule(doctor, set(new_patient_list))
                if not feasible:
                    
                    continue
                # is feasible
                schedules_n_patients[n].append((new_patient_list, objective_values))

        # print(len(schedules_n_patients[n]))
        # if we added no sets, break
        if not schedules_n_patients[n]:
            break

        n += 1

    all_tuple_schedules = []
    for n in schedules_n_patients:
        all_tuple_schedules.extend(schedules_n_patients[n])
    # all_tuple_schedules is in the form [([1], objs), ([1, 2], objs)]
    # all_tuple_schedules needs to be    {frozenset(1): objs, frozenset(1, 2): objs]
    schedules = {frozenset(tuple_schedule[0]): tuple_schedule[1] for tuple_schedule in all_tuple_schedules}
    return schedules


bad = gp.Model("rubbish")
print("-"*100)
# i,j = 5,1
# patients_j_can_treat = [i for k in diseases_doctor_qualified_for[j] for i in I_k[k] if compatible_times[i,j]]
# print("I_k[1]:", I_k[1])
# print(patients_j_can_treat)

# # print(doctor_times)
# print(doctor_times[j])
# print(patient_times[i])
# print(patient_diseases)
# print(patient_diseases[i])
# print(diseases_doctor_qualified_for[j])
# print("-")
# print(compatible_times[i,j])

S = dict()
for j in J:
    S[j] = find_all_patient_sets_for_doctor(j)
    print(len(S[j]))



<<<<<<< HEAD
=======
# ============================================================
# -------------------- Huge formulation ----------------------
# ============================================================
>>>>>>> Peleg

m = gp.Model("Doctor scheduling MIP")

# Doctor schedule
Z = {
    (j, s): m.addVar(vtype=gp.GRB.BINARY)
    for j in J for s in S
}

for obj in range(3):
    m.setObjective(gp.quicksum(s[obj] * Z[j, s] for j in J for s in S[j]), gp.GRB.MAXIMIZE)

    # Each patient is assigned at most once
    PatientsAreAssignedOnlyOnce = {
        i: m.addConstr(
            gp.quicksum(Z[j, s] for j in J for s in S[j] if i in s) <= 1
        )
        for i in I
    }

    # Each doctor has at most one schedule
    DoctorsHaveOnlyOneSchdeule = {
        s: m.addConstr(
            gp.quicksum(Z[j, s] for j in J) <= 1
        )
        for s in S[j]
    }

    m.optimize()

    # Zvals = {key: Z[key].x for key in Z}
    # Zs = {(j,s): Zvals.get((j,s), 0) for j in J for s in S[j]}
    # print_schedule()