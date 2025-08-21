import gurobipy as gp
import data_gen

K = range(2)
T = range(8)
J = range(4)
I = range(10)

disease_treat_times = data_gen.gen_diseases(K)
doctor_treat_times = data_gen.gen_doctor_time_treat(J, K, disease_treat_times)
doctor_can_treat = data_gen.gen_doctor_can_treat(doctor_treat_times, T)
doctor_disease_prefs = data_gen.gen_doctor_disease_preferences(doctor_can_treat)
doctor_availability = data_gen.gen_doctor_times(T, K, J, doctor_treat_times, doctor_can_treat)

print(doctor_treat_times)
print(doctor_can_treat)
print(doctor_disease_prefs)
print(doctor_availability)

patient_diseases = data_gen.gen_patient_diseases(I, K)
patient_doctor_prefs = data_gen.gen_patient_doctor_prefs(I, J, patient_diseases, doctor_can_treat)
patient_times = data_gen.gen_patient_times(I, J, T, patient_diseases, doctor_can_treat, doctor_treat_times)
        

print(patient_diseases)
print(patient_doctor_prefs)
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

m.setObjective(gp.quicksum(Y[i,j,t] for i in I for j in J for t in T), gp.GRB.MAXIMIZE)

m.optimize()
