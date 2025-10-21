import gurobipy as gp
from utils.data_gen import *
from src.utils.schedule_printing import *
from src.utils.logging_results import *
import pickle
import time
from typing import Dict, FrozenSet, Tuple, Optional

file = "data_seed10_I40_J4_K2_T20.pkl"
print("Using", file)
with open(file, "rb") as f:
    data = pickle.load(f)
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

MAX_FRAGMENT_LENGTH = 8

NEXT_AVAILABLE_TIME = 2
PATIENT_LIST = 1
START_TIME = 0

def patient_can_be_added_to_fragment(patient, doctor, fragment):
    fragment_time = fragment[NEXT_AVAILABLE_TIME] - fragment[START_TIME]

    # check that we have enough time left in the fragment
    if fragment_time + treat[doctor][patient_diseases[patient]] > MAX_FRAGMENT_LENGTH:
        return False
    if patient in fragment[PATIENT_LIST]:
        return False
    # return whether the patient can go in the next available time
    return fragment[NEXT_AVAILABLE_TIME] in compatible_times[patient, doctor]

def gen_fragments_for_doctor(doctor: int):

    fragments_length_n = dict()
    fragments_length_n[0] =  []
    patients = patients_doctor_can_treat[doctor]

    fragments_length_n[1] = [(start_time, (patient,), start_time + treat[doctor][patient_diseases[patient]] + 1) 
                             for patient in patients for start_time in compatible_times[patient, doctor]]
    print(1, len(fragments_length_n[1]))

    for n in range(2, MAX_FRAGMENT_LENGTH + 1):
        fragments_length_n[n] = []
        for fragment in fragments_length_n[n-1]:
            for patient in patients:
                if patient_can_be_added_to_fragment(patient, doctor, fragment):
                    fragments_length_n[n].append((fragment[START_TIME], 
                                                  fragment[PATIENT_LIST] + (patient,), 
                                                  fragment[NEXT_AVAILABLE_TIME] + treat[doctor][patient_diseases[patient]]))
        print(n, len(fragments_length_n[n]))
    
    allFragments = []
    for n in range(1, MAX_FRAGMENT_LENGTH + 1):
        allFragments.extend(fragments_length_n[n])
    return allFragments

F = dict()
for j in J:
    print(f"doctor: {j}, diseases: {diseases_doctor_qualified_for[j]}, treat times: {[treat[j][k] for k in diseases_doctor_qualified_for[j]]}, length available: {doctor_available[j][1]}, ")
    F[j] = gen_fragments_for_doctor(j)

m = gp.Model("Fragment")
# print((F[0]))
W = {j: {f: m.addVar(vtype=gp.GRB.BINARY) for f in F[j]} for j in J}

# Each patient is assigned at most once
PatientsAreAssignedOnlyOnce = {
    i: m.addConstr(
        gp.quicksum(W[j][f] for j in J for f in F[j] if i in f[PATIENT_LIST]) <= 1
    )
    for i in I
}

#fragments do not overlap
DoctorsAreNotOverbooked = \
{(j,t):
 m.addConstr(gp.quicksum(W[j][f] for f in W[j] if (f[START_TIME] <= t and t < f[NEXT_AVAILABLE_TIME])) <= 1)
 for j in J for t in T}

# -------------------- Objectives ----------------------------
# numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
# patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
# patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]
# fragment_patient_scores = {j: {f: 
#                                (sum(patientDoctorScore[i][j] + sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]
#                              for i in f[PATIENT_LIST])) for f in F[j]} for j in J}

# obj0 = gp.quicksum(Y[i,j,t] * 
#                            (
#                                 patientDoctorScore[i][j] 
#                                 + 
#                                 sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]
#                             )
#                            for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)


# obj0 = gp.quicksum

obj1 = gp.quicksum(W[j][f] * len(f[PATIENT_LIST]) for j in J for f in F[j])


doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
fragment_disease_scores = {j: {f: (sum(doctor_disease_rank_scores[j][patient_diseases[p]] for p in f[PATIENT_LIST])) for f in F[j]} for j in J}
obj2 = gp.quicksum(W[j][f] * fragment_disease_scores[j][f] for j in J for f in F[j])

objs = (0, obj1, obj2)
objectives = []
for obj_index in range(1,3):
    obj_lin_exp = objs[obj_index]
    m.setObjective(obj_lin_exp, gp.GRB.MAXIMIZE)

    m.setParam("OutputFlag", 0)
    m.optimize()

    objectives.append(round(m.ObjVal, 2))

    print("-" * 50)
    print("Maximise objective", obj_index)
    # schedule = create_schedule_from_Z(Z, S, J, T, treat, patient_diseases)
    # print_schedule_from_Z(schedule, I, J, T, doctor_times)

print("\n",objectives)