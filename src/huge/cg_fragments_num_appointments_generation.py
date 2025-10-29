import gurobipy as gp
# from data_gen import *
# from schedule_printing import *
# from logging_results import *
import pickle
import time
from typing import Dict, FrozenSet, Tuple, Optional
from utils.data_instance import DataInstance

# file = "data_seed10_I40_J4_K2_T20.pkl"
# print("Using", file)
# with open(file, "rb") as f:
#     data = pickle.load(f)
# globals().update(data)

# I = range(problem_size["patients"])
# J = range(problem_size["doctors"])
# K = range(problem_size["diseases"])
# T = range(problem_size["time periods"])

MAX_NUM_APPOINTMENTS = 4

NEXT_AVAILABLE_TIME = 2
PATIENT_LIST = 1
START_TIME = 0
PATIENT_TIME_LIST = 3

# def fragment_full(doctor, fragment):
#     min_treat_time = min(diseases_doctor_qualified_for[doctor])
#     return fragment[NEXT_AVAILABLE_TIME] - fragment[START_TIME] + min_treat_time > MAX_FRAGMENT_LENGTH

def patient_can_be_added_to_fragment(this_compat_times, patient, doctor, fragment):
    # fragment_time = fragment[NEXT_AVAILABLE_TIME] - fragment[START_TIME]
    # if j == 2:
    #     print(fragment, fragment_time)

    # check that we have enough time left in the fragment
    # if fragment_time + treat[doctor][patient_diseases[patient]] > MAX_FRAGMENT_LENGTH:
    #     # if j == 0 and patient not in fragment[PATIENT_LIST] and fragment[NEXT_AVAILABLE_TIME] in compatible_times[patient, doctor]:
    #     #     print("not enough time:", fragment_time + treat[doctor][patient_diseases[patient]])
    #     return False
    if patient in fragment[PATIENT_LIST]:
        return False
    # return whether the patient can go in the next available time
    # print("fragment:", fragment, "")
    return fragment[NEXT_AVAILABLE_TIME] in this_compat_times

def gen_fragments_for_doctor(d: DataInstance, doctor: int, max_frag_length: int):
    T = d.T
    fragments_length_n = dict()
    fragments_length_n[0] =  []
    patients = d.patients_doctor_can_treat[doctor]

    fragments_length_n[1] = [(start_time, (patient,), start_time + d.treat[doctor][d.patient_diseases[patient]], ((patient, start_time),)) 
                             for patient in patients for start_time in d.compatible_times[patient, doctor]]
    print(1, len(fragments_length_n[1]))
    max_length_fragments_grouped_by_next_available_time = {t: [] for t in T}
    fragments_by_start_time = {t: [] for t in T}
    for n in range(2, max_frag_length + 1):
        fragments_length_n[n] = []
        for fragment in fragments_length_n[n-1]:
            # if fragment_full(doctor, fragment):
                # if the fragment is full, stop
                # continue
            for patient in patients:
                if patient_can_be_added_to_fragment(d.compatible_times[patient, doctor], patient, doctor, fragment):
                    new_frag = (fragment[START_TIME], 
                                                  fragment[PATIENT_LIST] + (patient,), 
                                                  fragment[NEXT_AVAILABLE_TIME] + d.treat[doctor][d.patient_diseases[patient]],
                                                  fragment[PATIENT_TIME_LIST] + ((patient, fragment[NEXT_AVAILABLE_TIME]),)                                                  
                                                  )
                    fragments_by_start_time[new_frag[START_TIME]].append(new_frag)
                    if n == max_frag_length and new_frag[NEXT_AVAILABLE_TIME] < len(T):
                        max_length_fragments_grouped_by_next_available_time[new_frag[NEXT_AVAILABLE_TIME]].append(new_frag)

                    fragments_length_n[n].append(new_frag)
        print(n, len(fragments_length_n[n]))
    
    allFragments = []
    for n in range(1, max_frag_length + 1):
        allFragments.extend(fragments_length_n[n])
    # ff in F[j] if (len(ff[PATIENT_LIST]) == max_frag_length and ff[NEXT_AVAILABLE_TIME] == f[START_TIME]))
    return allFragments, max_length_fragments_grouped_by_next_available_time, fragments_by_start_time

def normal_generate_fragments(d: DataInstance, max_frag_length: int, save_output=True):
    F = dict()
    max_length_fragments_by_next_time = dict()
    fragments_by_start_time = dict()
    for j in d.J:
        print(f"doctor: {j}, diseases: {d.diseases_doctor_qualified_for[j]}, treat times: {[d.treat[j][k] for k in d.diseases_doctor_qualified_for[j]]}, length available: {d.doctor_available[j][1]}, ")
        F[j], max_length_fragments_by_next_time[j], fragments_by_start_time[j] = gen_fragments_for_doctor(d, j, max_frag_length)

    data = {
        "F": F,
        "max_length_fragments_by_next_time": max_length_fragments_by_next_time,
        "fragments_by_start_time": fragments_by_start_time,
        "I": d.I,
        "J": d.J,
        "T": d.T,
        "treat": d.treat,
        "patient_diseases": d.patient_diseases,
        "doctor_times": d.doctor_times,
    }

    if save_output:
        with open(f"data/cg_fragments_maxfraglength{max_frag_length}_seed{d.seed}_I{len(d.I)}_J{len(d.J)}_T{len(d.T)}_K{len(d.K)}.pkl", "wb") as f:
            pickle.dump(data, f)
    
    return data
