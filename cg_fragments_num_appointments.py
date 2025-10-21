import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
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

MAX_NUM_APPOINTMENTS = 4

NEXT_AVAILABLE_TIME = 2
PATIENT_LIST = 1
START_TIME = 0
PATIENT_TIME_LIST = 3

# def fragment_full(doctor, fragment):
#     min_treat_time = min(diseases_doctor_qualified_for[doctor])
#     return fragment[NEXT_AVAILABLE_TIME] - fragment[START_TIME] + min_treat_time > MAX_FRAGMENT_LENGTH

def patient_can_be_added_to_fragment(patient, doctor, fragment):
    # fragment_time = fragment[NEXT_AVAILABLE_TIME] - fragment[START_TIME]
    # if j == 2:
    #     print(fragment, fragment_time)

    # check that we have enough time left in the fragment
    # if fragment_time + treat[doctor][patient_diseases[patient]] > MAX_FRAGMENT_LENGTH:
    #     # if j == 0 and patient not in fragment[PATIENT_LIST] and fragment[NEXT_AVAILABLE_TIME] in compatible_times[patient, doctor]:
    #     #     print("not enough time:", fragment_time + treat[doctor][patient_diseases[patient]])
    #     return False
    if patient in fragment[PATIENT_LIST]:
        return False
    # return whether the patient can go in the next available time
    # print("fragment:", fragment, "")
    return fragment[NEXT_AVAILABLE_TIME] in compatible_times[patient, doctor]

def gen_fragments_for_doctor(doctor: int):

    fragments_length_n = dict()
    fragments_length_n[0] =  []
    patients = patients_doctor_can_treat[doctor]

    fragments_length_n[1] = [(start_time, (patient,), start_time + treat[doctor][patient_diseases[patient]], ((patient, start_time),)) 
                             for patient in patients for start_time in compatible_times[patient, doctor]]
    print(1, len(fragments_length_n[1]))
    max_length_fragments_grouped_by_next_available_time = {t: [] for t in T}
    fragments_by_start_time = {t: [] for t in T}
    for n in range(2, MAX_NUM_APPOINTMENTS + 1):
        fragments_length_n[n] = []
        for fragment in fragments_length_n[n-1]:
            # if fragment_full(doctor, fragment):
                # if the fragment is full, stop
                # continue
            for patient in patients:
                if patient_can_be_added_to_fragment(patient, doctor, fragment):
                    new_frag = (fragment[START_TIME], 
                                                  fragment[PATIENT_LIST] + (patient,), 
                                                  fragment[NEXT_AVAILABLE_TIME] + treat[doctor][patient_diseases[patient]],
                                                  fragment[PATIENT_TIME_LIST] + ((patient, fragment[NEXT_AVAILABLE_TIME]),)                                                  
                                                  )
                    fragments_by_start_time[new_frag[START_TIME]].append(new_frag)
                    if n == MAX_NUM_APPOINTMENTS:
                        max_length_fragments_grouped_by_next_available_time[new_frag[NEXT_AVAILABLE_TIME]].append(new_frag)

                    fragments_length_n[n].append(new_frag)
        print(n, len(fragments_length_n[n]))
    
    allFragments = []
    for n in range(1, MAX_NUM_APPOINTMENTS + 1):
        allFragments.extend(fragments_length_n[n])
    # ff in F[j] if (len(ff[PATIENT_LIST]) == MAX_NUM_APPOINTMENTS and ff[NEXT_AVAILABLE_TIME] == f[START_TIME]))
    return allFragments, max_length_fragments_grouped_by_next_available_time, fragments_by_start_time

F = dict()
max_length_fragments_by_next_time = dict()
fragments_by_start_time = dict()
for j in J:
    print(f"doctor: {j}, diseases: {diseases_doctor_qualified_for[j]}, treat times: {[treat[j][k] for k in diseases_doctor_qualified_for[j]]}, length available: {doctor_available[j][1]}, ")
    F[j], max_length_fragments_by_next_time[j], fragments_by_start_time[j] = gen_fragments_for_doctor(j)

m_start = time.perf_counter()
print("Starting to make model")
m = gp.Model("Fragment")
# print((F[0]))
W = {j: {f: m.addVar(vtype=gp.GRB.BINARY) for f in F[j]} for j in J}
print("Made W at ", time.perf_counter() - m_start)

# Each patient is assigned at most once
PatientsAreAssignedOnlyOnce = {
    i: m.addConstr(
        gp.quicksum(W[j][f] for j in J for f in F[j] if i in f[PATIENT_LIST]) <= 1
    )
    for i in I
}
print("Assign patients once", time.perf_counter() - m_start)


#fragments do not overlap
DoctorsAreNotOverbooked = \
{(j,t):
 m.addConstr(gp.quicksum(W[j][f] for f in W[j] if (f[START_TIME] <= t and t < f[NEXT_AVAILABLE_TIME])) <= 1)
 for j in J for t in T}
print("Doctors no overlap", time.perf_counter() - m_start)


# fragments come after no appointments or a full fragment
# B is 1 if there was just a break
B = {(j,t): m.addVar(vtype=gp.GRB.BINARY) for j in J for t in T[1:]}
for j in J:
    B[j, T[0]] = 1.0
print("Made Bs", time.perf_counter() - m_start)


SetBreaks = \
{(j,t):
 m.addConstr(B[j,t] == B[j, t-1] 
             - gp.quicksum(W[j][f] for f in F[j] if f[START_TIME] == t - 1) # start in the previous time period
             + gp.quicksum(W[j][f] for f in F[j] if f[NEXT_AVAILABLE_TIME] == t - 1) # ended in the previous time period
              )
 for j in J for t in T[1:]}
print("Set break flow", time.perf_counter() - m_start)

SymmetryBreak = \
{(j, t):
 # W[j][f] can only be on if the previous fragment was 3 or it is ont a break
 m.addConstr(gp.quicksum(W[j][f] for f in fragments_by_start_time[j][t]) <= 
             # a previous group of 3 appointments
             gp.quicksum(W[j][f] for f in max_length_fragments_by_next_time[j][t])
             + B[j,t]
             )
 for j in J for t in T
}
print("Break symmetry using breaks", time.perf_counter() - m_start)

# -------------------- Objectives ----------------------------
numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]
fragment_patient_scores = {j: {f: 
                               (sum(
                                   patientDoctorScore[i][j] 
                                   + 
                                   sum(patientTimeScore[i][t:min(t + treat[j][patient_diseases[i]], len(T))]) / treat[j][patient_diseases[i]]
                             for i,t in f[PATIENT_TIME_LIST])) for f in F[j]} for j in J}


obj0 = gp.quicksum(W[j][f] * fragment_patient_scores[j][f] for j in J for f in F[j])


obj1 = gp.quicksum(W[j][f] * len(f[PATIENT_LIST]) for j in J for f in F[j])


doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
fragment_disease_scores = {j: {f: (sum(doctor_disease_rank_scores[j][patient_diseases[p]] for p in f[PATIENT_LIST])) for f in F[j]} for j in J}
obj2 = gp.quicksum(W[j][f] * fragment_disease_scores[j][f] for j in J for f in F[j])

objs = (obj0, obj1, obj2)
objectives = []
for obj_index in range(0,3):
    obj_lin_exp = objs[obj_index]
    m.setObjective(obj_lin_exp, gp.GRB.MAXIMIZE)

    m.setParam("OutputFlag", 1)
    m.optimize()

    objectives.append(round(m.ObjVal, 3))

    print("-" * 50)
    print("Maximise objective", obj_index)
    # schedule = create_schedule_from_Z(Z, S, J, T, treat, patient_diseases)
    # print_schedule_from_Z(schedule, I, J, T, doctor_times)

print("\n",objectives)

# def optimise_and_print_schedule_from_W_fragments(m, M1, W, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times):
#     start_time = time.perf_counter()
#     m.optimize()
#     print("optimising time:", time.perf_counter() - start_time)
    
    
#     Yvals = {key: Y[key].x for key in Y}
#     Ys = {(i,j,t): Yvals.get((i,j,t), 0) for i in I for j in J for t in T}

#     schedule = create_schedule(Ys, K, J, I_k, T, treat)
#     print_stats(Ys, M1, I, J, K, T, I_k, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs)
#     print_schedule(schedule, I, J, T, doctor_times)
#     plot_schedule(schedule, I, J, T, doctor_times, path="plot.png")