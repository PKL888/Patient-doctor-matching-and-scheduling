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
def find_all_patient_sets_for_doctor(doctor: int) -> Dict[FrozenSet[int], Tuple[float, float, float]]:
    patients = patients_doctor_can_treat[j]

    schedules_n_patients = dict()
    #                             (set with no patients, tuple of 0 objective values)
    #                         change to not be sets????
    schedules_n_patients[0] = [ ([], (0,0,0)) ]
    schedules_n_patients[1] = [([patient], find_best_schedule(doctor, {patient})[2]) for patient in patients]

    n = 2
    while True:
        schedules_n_patients[n] = []
        for patient_list, _ in schedules_n_patients[n - 1]:
            
            # {([1], (obj0, obj1, obj2)), ([2], (obj0, obj1, obj2))}
            # schedule is ([1], (obj0, obj1, obj2))

            last_patient = patient_list[-1]
            potential_patients = I[last_patient + 1:]
            for patient in potential_patients:
                new_patient_list = patient_list + [patient]
                                                                           # make take in a list???
                feasible, _, objective_values = find_best_schedule(doctor, set(new_patient_list))
                if not feasible:
                    continue
                # is feasible
                schedules_n_patients[n].append((new_patient_list, objective_values))

        # if we added no sets, break
        if not schedules_n_patients[n]:
            break

        n += 1

    all_tuple_schedules = []
    for n in schedules_n_patients:
        all_tuple_schedules.extend(schedules_n_patients[n])
    # all_tuple_schedules is in the form [([1], objs), ([1, 2], objs)]
    # all_tuple_schedules needs to be    {frozenset(1): objs, frozenset(1, 2): objs]
    schedules = {frozenset(tuple_schedule[0]): tuple_schedule[1] for tuple_schedule in all_tuple_schedules}
    return schedules




S = dict()
for j in J:
    S[j] = find_all_patient_sets_for_doctor(j)




"""
Finds the best schedule for these patients for this doctor

If it is infeasible, the objective values are not set.

Returns: (true, paitents, (objective_value 1, val 2, val 3))
"""
def find_best_schedule(doctor: int, patients:set[int]) -> tuple[bool, set[int], tuple[float]]:
    pass

# set of schedules


# mip







m = gp.Model("Doctor availability")