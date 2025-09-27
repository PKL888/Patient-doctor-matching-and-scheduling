import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import pickle

# Load saved schedules
with open("cg_output.pkl", "rb") as f:
    data = pickle.load(f)
globals().update(data)

# ============================================================
# -------------------- Huge formulation ----------------------
# ============================================================

m = gp.Model("Doctor scheduling MIP")

# Doctor schedule
Z = {
    (j, s): m.addVar(vtype=gp.GRB.BINARY)
    for j in J for s in S[j]
}

# Each patient is assigned at most once
PatientsAreAssignedOnlyOnce = {
    i: m.addConstr(
        gp.quicksum(Z[j, s] for j in J for s in S[j] if i in s) <= 1
    )
    for i in I
}

# Each doctor has at most one schedule
DoctorsHaveOnlyOneSchdeule = {
    j: m.addConstr(
        gp.quicksum(Z[j, s] for s in S[j]) == 1
    )
    for j in J
}

objectives = []
for obj in range(3):
    m.setObjective(gp.quicksum(S[j][s][0][obj] * Z[j, s] for j in J for s in S[j]), gp.GRB.MAXIMIZE)

    m.setParam("OutputFlag", 0)
    m.optimize()

    objectives.append(round(m.ObjVal, 2))

    print("-" * 50)
    print("Maximise objective", obj)
    schedule = create_schedule_from_Z(Z, S, J, T, treat, patient_diseases)
    print_schedule_from_Z(schedule, I, J, T, doctor_times)

print("\n",objectives)