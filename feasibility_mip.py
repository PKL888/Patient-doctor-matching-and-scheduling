import gurobipy as gp
from data_gen import *

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

m = gp.Model("Doctor-patient feasibility")

# Variables
Y = {(i,j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for i in I for j in J for t in T}

# Constraints
PatientsAreSeenAtMostOnce = \
{i:
 m.addConstr(gp.quicksum(Y[i,j,t] for j in J for t in T) <= 1)
 for i in I}

DoctorsAreNotOverbooked = \
{(j,t):
 m.addConstr(gp.quicksum(Y[i,j,tt] for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t]) <= 1)
 for j in J for t in T}

FeasibleTime = \
{(i,j,k,t):
 m.addConstr(Y[i,j,t] <= doctor_available[j,tt] * patient_available[i,tt])
 for k in K for i in I_k[k] for j in J for t in T for tt in range(t, min(t + treat[j][k] - 1, T[-1] + 1))}

m.setObjective(gp.quicksum(Y[i,j,t] for i in I for j in J for t in T), gp.GRB.MAXIMIZE)

m.optimize()
