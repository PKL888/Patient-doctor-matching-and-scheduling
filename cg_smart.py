import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import random
import pickle
import json
import time

from typing import Dict, FrozenSet, Tuple


with open("data_seed10_I100_J10_K4_T20.pkl", "rb") as f:
    data = pickle.load(f)

# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

START = 0
DURATION = 1

"""
Finds all the patient

Returns: A set of sets of patients for this doctor, with the scores for each objective

"""
def find_all_patient_sets_for_doctor(doctor: int) -> Dict[FrozenSet[int], Tuple[int]]:
    pass 

S = dict()
for j in J:
    S[j] = find_all_patient_sets_for_doctor(j)




"""
Finds the best schedule for these patients for this doctor

If it is infeasible, the objective values are not set.

Returns: (true, paitents, (objective_value 1, val 2, val 3))
"""
def find_best_schedule(doctor: int, patients:set{int}) -> tuple(bool, set{int}, tuple(float)):
    pass

# ============================================================
# -------------------- Huge formulation ----------------------
# ============================================================

m = gp.Model("Doctor scheduling MIP")

# Doctor schedule
Z = {
    (j, s): m.addVar(vtype=gp.GRB.BINARY)
    for j in J for s in S
}

for obj in range(3):
    m.setObjective(gp.quicksum(s[obj] * Z[j, s] for j in J for s in S[j]), gp.GRB.MAXIMIZE)

    # Each patient is assigned at most once
    PatientsAreAssignedOnlyOnce = {
        i: m.addConstr(
            gp.quicksum(Z[j, s] for j in J for s in S[j] if i in s) <= 1
        )
        for i in I
    }

    # Each doctor has at most one schedule
    DoctorsHaveOnlyOneSchdeule = {
        s: m.addConstr(
            gp.quicksum(Z[j, s] for j in J) <= 1
        )
        for s in S[j]
    }

    m.optimize()

    # Zvals = {key: Z[key].x for key in Z}
    # Zs = {(j,s): Zvals.get((j,s), 0) for j in J for s in S[j]}
    # print_schedule()