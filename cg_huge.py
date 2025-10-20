import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import pickle
import time

def do_huge(data_name):
    # Load saved schedules
    with open(data_name, "rb") as f:
        data = pickle.load(f)
    globals().update(data)
    print(sum(len(S[j]) for j in J))

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
        start_time = time.perf_counter()
        m.setObjective(gp.quicksum(S[j][s][0][obj] * Z[j, s] for j in J for s in S[j]), gp.GRB.MAXIMIZE)

        m.setParam("OutputFlag", 0)
        m.optimize()

        objectives.append(round(m.ObjVal, 2))

        print("-" * 50)
        print("Maximise objective", obj)
        schedule = create_schedule_from_Z(Z, S, J, T, treat, patient_diseases)
        print_schedule_from_Z(schedule, I, J, T, doctor_times)
        print("time taken:", time.perf_counter()- start_time)

    print("\n",objectives)

data_names = ["cg_smart_output_I5_J5_T20_K2.pkl","cg_smart_output_I10_J5_T20_K2.pkl","cg_smart_output_I15_J5_T20_K2.pkl","cg_smart_output_I20_J5_T20_K2.pkl"]

# for data_name in data_names[:1]:
#     do_huge(data_name)
do_huge("cg_smart_output_I25_J5_T20_K2.pkl")