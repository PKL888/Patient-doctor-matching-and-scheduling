import gurobipy as gp
import data_gen
import random

random.seed(10)

I = range(10)
J = range(4)
K = range(2)
T = range(8)

disease_treat_times = data_gen.gen_diseases(K)
treat = data_gen.gen_doctor_time_treat(J, K, disease_treat_times)
doctor_can_treat = data_gen.gen_doctor_can_treat(treat, T)
doctor_disease_prefs = data_gen.gen_doctor_disease_preferences(doctor_can_treat)
doctor_times = data_gen.gen_doctor_times(T, K, J, treat, doctor_can_treat)

# create a binary list for doctor availability per doctor per time period
doctor_availability = []
for d in doctor_times:
    time_list = []
    binary_list = []
    for j in range(d[0], d[0]+d[1]):
        time_list.append(j)
    for i in range(8):
        if i in time_list:
            binary_list.append(1)
        else:
            binary_list.append(0)
    doctor_availability.append(binary_list)


print(treat)
print(doctor_can_treat)
print(doctor_disease_prefs)
print("available")
print(doctor_availability)

patient_diseases = data_gen.gen_patient_diseases(I, K)

I_k = [[i for i in I if patient_diseases[i] == k] for k in K]

patient_doctor_prefs = data_gen.gen_patient_doctor_prefs(I, J, patient_diseases, doctor_can_treat)
patient_availability = data_gen.gen_patient_times(I, J, T, patient_diseases, doctor_can_treat, treat)

# create a binary list for patient availability per patient per time period 
patient_times = []
for p in patient_availability:
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

print(patient_diseases)
print(patient_doctor_prefs)
print("available")
print(patient_times)

m = gp.Model("Doctor patient feasibility")

# variables
Y = {(i,j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for i in I for j in J for t in T}

# constraints
AllPatientsAreSeenAtMostOnce = \
{i:
m.addConstr(gp.quicksum(Y[i,j,t] for j in J for t in T) <= 1)
for i in I}

DoctorsAreNotOverbooked = \
{(j,t):
m.addConstr(gp.quicksum(Y[i,j,tt] for k in K for i in I_k[k] for tt in T[max(0, t-treat[j][k] + 1):t]) <= 1)
for j in J for t in T}

FeasibleTime = \
{(i,j,k,t):
 m.addConstr(Y[i,j,t] <= doctor_availability[j][tt] * patient_times[i][tt])
 for k in K for i in I_k[k] for j in J for t in T for tt in range(t, min(t + treat[j][k] - 1, T[-1] + 1))}

m.setObjective(gp.quicksum(Y[i,j,t] for i in I for j in J for t in T), gp.GRB.MAXIMIZE)

m.optimize()
