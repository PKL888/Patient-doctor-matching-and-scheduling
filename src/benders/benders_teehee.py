import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import gurobipy as gp
from src.utils.data_gen import *
# from src.utils.print_model_results import *
# from src.utils.logging_results import *
import random
import pickle
import json
import time
# NOT INCLUDED IN REPORT
# with open("data/data_seed10_I100_J10_K4_T20.pkl", "rb") as f:
#     data = pickle.load(f)
problem_size = {
        "patients": 100,
        "doctors":  10,
        "diseases": 4,
        "time periods": 20
    }
all_data = get_data(problem_size, seeds = [0])
data = all_data[f"seed_{0}"]


# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

m = gp.Model("Doctor availability")

# start time for pre-solver
start_time = time.time()

##################################################################################
# Variables
U = {(i,j):
    m.addVar(vtype=gp.GRB.BINARY)
    for j in J for i in patients_doctor_can_treat[j]
    }

##################################################################################
# Constraints
PatientsAreAssignedOnlyOnce = \
{(i):
 m.addConstr(gp.quicksum(U[ii,j] for (ii,j) in U if (ii == i)) <=1) # FIX LATER
for i in I
}

DoctorsScheduledTimeIsLessThanAvailable = \
{j:
    m.addConstr(gp.quicksum(U[i,j] * treat[j][patient_diseases[i]] for i in patients_doctor_can_treat[j]) <= doctor_available[j][1])
    for j in J
}

def make_small_mip_model_compatible_times(doctor:int, patients: frozenset[int]):
    m = gp.Model("Small MIP")
    m.setParam("OutputFlag", 0)

    # -------------------- Variables -----------------------------
    Y = {
        (i,doctor,t): m.addVar(vtype=gp.GRB.BINARY)
        for i in patients for t in compatible_times[i,doctor]
    }

    B = {t:
         m.addVar(vtype=gp.GRB.BINARY)
         for t in T}

    # -------------------- Constraints ---------------------------
    for i in patients:
        m.addConstr(gp.quicksum(Y[i, doctor, t] for t in compatible_times[i, doctor]) == 1)

    OneBIsOn = m.addConstr(gp.quicksum(B[t] for t in B) >= 1)

    DoctorIsOverbookedmwaHaHAHAHA = {
        t: m.addConstr(
            gp.quicksum(Y[i,doctor,tt] 
            for i in patients for tt in T[
                max(0, t - treat[doctor][patient_diseases[i]] + 1):
                t+1
                ] 
            if tt in compatible_times[i, doctor]) 
            >= 2 * B[t]
        )
    for t in T}

    # minimise number of patients in set with less
    # WANT MINIMAL SET WHERE IT CANNOT BE SCHEDULED - CAN ONLY PICK EACH PATIENT ONCE.
    # MUST FAIL?
    m.setObjective(gp.quicksum(Y[i,j,t] for (i,j,t) in Y))

    return m, Y

sols = [{frozenset(): True} for j in J]
def Callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # print("In callback")
        UV = model.cbGetSolution(U)

        # for each doctor, check you can make a schedule
        for j in J:
            # check there is a schedule
            patients = frozenset(i for i in patients_doctor_can_treat[j] if round(UV[i,j]))
            for prev_checked_patients in sols[j]:
                if patients <= prev_checked_patients:
                    if sols[j][prev_checked_patients]:
                        continue
            # if patients in sols[j]:
            #     if sols[j][patients]:
            #         continue
                # else:
                    # print(f"ERROR: patients {patients} not cut off for doctor {j}")
                    # break
            
            BSP, Y = make_small_mip_model_compatible_times(j, patients)

            BSP.optimize()
            sols[j][patients] = (not BSP.status == gp.GRB.OPTIMAL)
            if BSP.status == gp.GRB.OPTIMAL:
                # 
                limitingPatients = [i for (i,j,t) in Y if round(Y[i,j,t].x)]
                # cut off solution
                # print(f"Cutting off patients {patients} for doctor {j}")
                m.cbLazy(gp.quicksum(U[i,j] for i in limitingPatients) <= len(limitingPatients) - 1)
                m.cbLazy(gp.quicksum(U[i,j] for i in patients) <= len(patients) - 1) # does this do nothing with previous constraint

            # IDEAS:
            # if there is a solution, cache it?
            # if we are optimising for # of appointments, then if a set is feasible, then all subsets are feasible
            # if we are optimising for pat sat, we need to rerun it for the subset

#################################################################
# printing and optimising

# m.setParam("OutputFlag", 1)
m.setObjective(gp.quicksum(U[i,j] for (i,j) in U), gp.GRB.MAXIMIZE)
m.Params.LazyConstraints = 1
m.optimize(Callback)

for j in J:
    print(j, [i for i in patients_doctor_can_treat[j] if round(U[i,j].x)])
