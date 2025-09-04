import gurobipy as gp
from data_gen import *
import random
import json

with open("data_seed10_I100_J10_K4_T20.json", "r") as f:
    data = json.load(f)



# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = [t for t in range(problem_size["time periods"])]

START = 0
DURATION = 1

I_k = [[i for i in I if patient_diseases[i] == k] for k in K]
J_k = [[j for j in J if qualified[j][k]] for k in K]

diseases_doctor_qualified_for = {j: [k for k in K if qualified[j][k]] for j in J}

# diseases_doctor_can_treat_at_time = {(j,t): [k for k in diseases_doctor_qualified_for[j] if t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] + 1]]
#                                      for j in J for t in T}

# patients_with_disease_that_can_be_treated_at_time = \
# {(k,t):
#     [i for i in I_k[k] if t in T[patient_available[i][START]: patient_available[i][START] + patient_available[i][DURATION]]] 
# for k in K for t in T}

compatible_times = {(i,j):
                    T[max(patient_available[i][START], doctor_available[j][START]):
      (max(0, min(patient_available[i][START] + patient_available[i][DURATION], doctor_available[j][START] + doctor_available[j][DURATION]) - treat[j][k] + 1))]
      for k in K for i in I_k[k] for j in J_k[k]}

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

m = gp.Model("Doctor availability")

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
def left_pad_string(s, length):
    if len(s) >= length:
        return s
    
    return " " * (length - len(s)) + s 

def create_schedule(Ys):
    schedule = []
    for j in J:
        # print("doctor:", j, "treatment length:", treat[j])
        # print("times available", doctor_times[j])
        # print("start appointment", [sum(Y[i,j,t].x for i in I) for t in T])
        # print("checking ", [sum((Y[i,j,tt].x) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) and not doctor_times[j][t] for t in T])
        doctor_schedule = [int(patient - 1) for patient in [sum((Ys[i,j,tt] * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]]
        # doctor_schedule_with_disease = [(patient, patient_diseases[patient]) for patient in doctor_schedule]
        # print("(patient, disease)", [(int(patient - 1), patient_diseases[int(patient - 1)]) for patient in [sum((Y[i,j,tt].x * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]])
        # print("(patient, disease)", doctor_schedule_with_disease)
        schedule.append(doctor_schedule)
    return schedule

def print_stats(Ys):
    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
    print("Stats: -----------------------------------")
    print("Number of patients allocated:", round(sum(Ys[i,j,t] for i in I for j in J for t in T)))
    print("Patient satisfaction with doctor and time:", round(sum(Ys[i,j,t] * ((numberAvailableDoctors[i] - allocate_rank[i][j] + 1)/numberAvailableDoctors[i] + ((patient_available[i][1]) + 1 - patient_time_prefs[i][t])/patient_available[i][1]) for i in I for j in J for t in T)))
    print("Doctor satisfaction with diseases:", round(sum((doctor_disease_rank_scores[j][k]) * Ys[i,j,t] for k in K for i in I_k[k] for j in J for t in T)))
    print("Appointments per doctor:", round(sum(Ys[i,j,t] for i in I for j in J for t in T))/len(J))

def print_schedule(schedule):
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))
    for j in J:
        formatted_doctor_schedule = [(patient >= 0) * str(patient) + (patient < 0) * " " + "-" * (1 - doctor_times[j][t]) for t, patient in enumerate(schedule[j])]
        padded_doctor_shedule = [left_pad_string(s, padding) for s in formatted_doctor_schedule]
        print("doctor:", j, " ".join(padded_doctor_shedule))

def optimise_and_print_schedule():
    m.optimize()
    Yvals = {key: Y[key].x for key in Y}
    Ys = {(i,j,t): Yvals.get((i,j,t), 0) for i in I for j in J for t in T}

    schedule = create_schedule(Ys)
    print_stats(Ys)
    print_schedule(schedule)

# m.setParam("OutputFlag", 0)

# # Objective 1: Max. number of matches
# print("Objective 1: Max. number of matches")

# m.setObjective(gp.quicksum(Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
# optimise_and_print_schedule()

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
optimise_and_print_schedule()
# m.optimize()


# # Objective 3: Max. doctor satisfaction
# print("Objective 3: Max. doctor satisfaction")

# doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
# doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

# m.setObjective(gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
# optimise_and_print_schedule()