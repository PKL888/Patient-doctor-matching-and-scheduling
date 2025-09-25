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

# set of schedules


# mip







m = gp.Model("Doctor availability")