import gurobipy as gp
from data_gen import *
import random

random.seed(10)

I = range(100)
J = range(10)
K = range(5)
T = range(20)

best = gen_best(K)
treat = gen_treat(J, K, best)
qualified = gen_qualified(T, treat)

# print("Best treatment times:", best)
print("Doctor service times:", treat[0][2])
# print("Enough time to treat:", qualified)

doctor_rank = gen_doctor_rank(qualified)
doctor_available = gen_doctor_available(J, K, T, qualified, treat)

# print("-" * 21)
# print("Disease rank by doct:", doctor_rank)
# print("Doctor start, length:", doctor_available)

patient_diseases = gen_patient_diseases(I, K)
allocate_rank = gen_allocate_rank(I, J, patient_diseases, qualified)
patient_available = gen_patient_available(I, J, T, patient_diseases, qualified, treat)
patient_time_prefs = gen_patient_time_prefs(I, T, patient_available)

# print("-" * 21)
# print("Diseases by patients:", patient_diseases)
# print("Doctor rank by patie:", allocate_rank)
# print("Patient start, lengt:", patient_available)
# print("Patient time prefere:", patient_time_prefs)

I_k = [[i for i in I if patient_diseases[i] == k] for k in K]

# Create a binary list for doctor availability per doctor per time period
doctor_times = []
for d in doctor_available:
    time_list = []
    binary_list = []
    for j in range(d[0], d[0]+d[1]):
        time_list.append(j)
    for i in range(len(T)):
        if i in time_list:
            binary_list.append(1)
        else:
            binary_list.append(0)
    doctor_times.append(binary_list)

# Create a binary list for patient availability per patient per time period 
patient_times = []
for p in patient_available:
    time_list = []
    binary_list = []
    for j in range(p[0], p[0]+p[1]):
        time_list.append(j)
    for i in range(len(T)):
        if i in time_list:
            binary_list.append(1)
        else:
            binary_list.append(0)
    patient_times.append(binary_list) 

m = gp.Model("Doctor patient feasibility")

# Variables
Y = {(i,j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for i in I for j in J for t in T}

# Constraints
DoctorsAreNotOverbooked = \
{(j,t):
 m.addConstr(gp.quicksum(Y[i,j,tt] for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) <= 1)
 for j in J for t in T}

PatientsAreSeenAtMostOnce = \
{i:
 m.addConstr(gp.quicksum(Y[i,j,t] for j in J for t in T) <= 1)
 for i in I}

FeasibleTime = \
{(j,k,t):
 m.addConstr(treat[j][k] * Y[i,j,t] <= sum(doctor_times[j][tt] * patient_times[i][tt] for tt in range(t, min(t + treat[j][k], len(T)))))
 for k in K for i in I_k[k] for j in J for t in T}

DoctorsQualified = \
{(i,j,k,t):
m.addConstr( Y[i,j,t] <= qualified[j][k])
for k in K for i in I_k[k] for j in J for t in T}

def left_pad_string(s, length):
    if len(s) >= length:
        return s
    
    return " " * (3 - len(s)) + s 

def create_schedule():
    schedule = []
    for j in J:
        # print("doctor:", j, "treatment length:", treat[j])
        # print("times available", doctor_times[j])
        # print("start appointment", [sum(Y[i,j,t].x for i in I) for t in T])
        # print("checking ", [sum((Y[i,j,tt].x) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) and not doctor_times[j][t] for t in T])
        doctor_schedule = [int(patient - 1) for patient in [sum((Y[i,j,tt].x * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]]
        doctor_schedule_with_disease = [(patient, patient_diseases[patient]) for patient in doctor_schedule]
        # print("(patient, disease)", [(int(patient - 1), patient_diseases[int(patient - 1)]) for patient in [sum((Y[i,j,tt].x * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]])
        # print("(patient, disease)", doctor_schedule_with_disease)
        schedule.append(doctor_schedule)
    return schedule

def print_stats():
    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
    print("Stats: -----------------------------------")
    print("Number of patients allocated:", round(sum(Y[i,j,t].x for i in I for j in J for t in T)))
    print("Doctor satisfaction with diseases:", round(sum((doctor_disease_rank_scores[j][k]) * Y[i,j,t].x for k in K for i in I_k[k] for j in J for t in T)))
    print("Patient satisfaction with doctor and time:", round(sum(Y[i,j,t].x * ((numberAvailableDoctors[i] - allocate_rank[i][j] + 1)/numberAvailableDoctors[i] + ((patient_available[i][1]) + 1 - patient_time_prefs[i][t])/patient_available[i][1]) for i in I for j in J for t in T)))
    print("Appointments per doctor:", round(sum(Y[i,j,t].x for i in I for j in J for t in T))/len(J))

def print_schedule(schedule):
    padding = 3
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))
    for j in J:
        formatted_doctor_schedule = [(patient >= 0) * str(patient) + (patient < 0) * " " + "-" * (1 - doctor_times[j][t]) for t, patient in enumerate(schedule[j])]
        padded_doctor_shedule = [left_pad_string(s, padding) for s in formatted_doctor_schedule]
        print("doctor:", j, " ".join(padded_doctor_shedule))

def optimise_and_print_schedule():
    m.optimize()

    schedule = create_schedule()
    print_stats()
    print_schedule(schedule)

m.setParam("OutputFlag", 0)

# Objective 1: Max. number of matches
print("Objective 1: Max. number of matches")

m.setObjective(gp.quicksum(Y[i,j,t] for i in I for j in J for t in T), gp.GRB.MAXIMIZE)
optimise_and_print_schedule()

# Objective 2: Max. patient satisfaction
print("Objective 2: Max. patient satisfaction")

numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]

m.setObjective(gp.quicksum(Y[i,j,t] * (patientDoctorScore[i][j] + 
                                       sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / 
                                       treat[j][k])
                           for k in K for i in I_k[k] for j in J for t in T), gp.GRB.MAXIMIZE)
optimise_and_print_schedule()

# Objective 3: Max. doctor satisfaction
print("Objective 3: Max. doctor satisfaction")

doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

m.setObjective(gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J for t in T), gp.GRB.MAXIMIZE)
optimise_and_print_schedule()