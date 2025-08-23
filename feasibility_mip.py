import gurobipy as gp
from data_gen import *
import random

random.seed(10)

I = range(10)
J = range(4)
K = range(2)
T = range(8)

random.seed(10)

best = gen_best(K)
treat = gen_treat(J, K, best)
qualified = gen_qualified(T, treat)

print("Best treatment times:", best)
print("Doctor service times:",treat)
print("Enough time to treat:", qualified)

doctor_rank = gen_doctor_rank(qualified)
doctor_available = gen_doctor_available(J, K, T, qualified, treat)

print("-" * 21)
print("Disease rank by doct:", doctor_rank)
print("Doctor start, length:", doctor_available)

patient_diseases = gen_patient_diseases(I, K)
allocate_rank = gen_allocate_rank(I, J, patient_diseases, qualified)
patient_available = gen_patient_available(I, J, T, patient_diseases, qualified, treat)

print("-" * 21)
print("Diseases by patients:", patient_diseases)
print("Doctor rank by patie:", allocate_rank)
print("Patient start, lengt:", patient_available)

I_k = [[i for i in I if patient_diseases[i] == k] for k in K]

# Create a binary list for doctor availability per doctor per time period
doctor_times = []
for d in doctor_available:
    time_list = []
    binary_list = []
    for j in range(d[0], d[0]+d[1]):
        time_list.append(j)
    for i in range(8):
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
    for i in range(8):
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
 m.addConstr(Y[i,j,t] <= doctor_times[j][tt] * patient_times[i][tt])
 for k in K for i in I_k[k] for j in J for t in T for tt in range(t, min(t + treat[j][k] + 1, T[-1] + 1))}

m.setObjective(gp.quicksum(Y[i,j,t] for i in I for j in J for t in T), gp.GRB.MAXIMIZE)

m.optimize()

for j in J:
    print("doctor:", j, "treatment length:", treat[j])
    print("times available", doctor_times[j])
    print("start appointment", [sum(Y[i,j,t].x for i in I) for t in T])
    # print("checking ", [sum((Y[i,j,tt].x) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) and not doctor_times[j][t] for t in T])
    print("(patient, disease)", [(int(patient - 1), patient_diseases[int(patient - 1)]) for patient in [sum((Y[i,j,tt].x * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]])
    
