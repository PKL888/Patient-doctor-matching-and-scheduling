import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import pickle

from typing import Dict, FrozenSet, Tuple, Optional

with open("data_seed10_I20_J4_K2_T10.pkl", "rb") as f:
    data = pickle.load(f)
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

START = 0
DURATION = 1

"""
Find the best schedule for a given set of patients, for a specific doctor, by solving a small MIP

If no schedule is possible (i.e., the doctor cannot treat all the patients in the set), then returns false

Returns: (true, patients, objective values)
"""
def find_best_schedule(doctor: int, patients:set[int]) -> Tuple[bool, Optional[Dict[Tuple[int, int, int], int]], Optional[Tuple[float, float, float]]]:
    m = gp.Model("Small MIP")
    m.setParam("OutputFlag", 0)

    # -------------------- Variables -----------------------------
    
    # Y[i,doctor,t] = 1 if patient i is assigned to the doctor starting at time t
    Y = {
        (i,doctor,t): m.addVar(vtype=gp.GRB.BINARY)
        for i in patients for t in compatible_times[i,doctor]
    }

    # Z[doctor,t] = 1 if the doctor is available at time t
    Z = {
        (doctor,t): m.addVar(vtype=gp.GRB.BINARY)
        for t in T[
            doctor_available[doctor][START]:
            doctor_available[doctor][START] + doctor_available[doctor][DURATION]
        ]
    }
    
    # -------------------- Constraints ---------------------------

    # Each patient is assigned at most once
    AllPatientsSeenOnce = {
        i: m.addConstr(
            gp.quicksum(
                Y[i, doctor, t] 
                for t in compatible_times[i, doctor]
            ) == 1
        ) 
     for i in patients
    }
    
    # Doctor availability dynamics
    DoctorAvailableConstraint = {
        (doctor,t): m.addConstr(Z[doctor,t] == Z[doctor,t-1]
            + gp.quicksum(
                Y[i,doctor,t-treat[doctor][patient_diseases[i]]]
                for i in patients 
                if t-treat[doctor][patient_diseases[i]] in compatible_times[i,doctor]
            )
            - gp.quicksum(
                Y[i,doctor,t] 
                for i in patients 
                if t in compatible_times[i,doctor]
            ) # outgoing
        )
        for t in T[
            doctor_available[doctor][START] + 1:
            doctor_available[doctor][START] + doctor_available[doctor][DURATION]
        ]
    }

    # Initial doctor availability
    DoctorStartsAvailable = m.addConstr(
        Z[doctor,doctor_available[doctor][START]] == 1 
        - gp.quicksum(
            Y[i,doctor,doctor_available[doctor][START]] 
            for i in patients 
            if doctor_available[doctor][START] in compatible_times[i,doctor]
        )
    )

    # Final doctor availability
    DoctorEndsAvailable = m.addConstr(
        Z[doctor,doctor_available[doctor][START] + doctor_available[doctor][DURATION]-1]
        + gp.quicksum(
            Y[i,doctor,doctor_available[doctor][START] + doctor_available[doctor][DURATION] 
            - treat[doctor][patient_diseases[i]]] 
            for i in patients 
            if doctor_available[doctor][START] + doctor_available[doctor][DURATION] 
            - treat[doctor][patient_diseases[i]] in compatible_times[i,doctor]
        ) == 1
    )

    # All patients are assigned 
    # m.addConstr(gp.quicksum(Y[i, doctor, t] for i in patients for t in compatible_times[i,doctor]) == len(patients))

    # -------------------- Objectives ----------------------------

    # ---------- Patient-doctor score ----------
    # Number of doctors available to each patient
    numberAvailableDoctors = {
        i: sum(allocate_rank[i][jj] != M1 for jj in J)
        for i in patients
    }

    # Patient-doctor score for the current doctor only
    patientDoctorScore = {
        i: ((numberAvailableDoctors[i] - allocate_rank[i][doctor] + 1) / numberAvailableDoctors[i] if allocate_rank[i][doctor] != M1 else 0)
        for i in patients
    }

    # ---------- Patient-time score ----------
    patientTimeScore = {
        i: [(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] 
        for i in patients
    }

    # Objective 0: Patient satisfaction
    objective_0 = gp.quicksum(
        Y[i,doctor,t] * (
            patientDoctorScore[i]
            + sum(patientTimeScore[i][t:min(t + treat[doctor][patient_diseases[i]], len(T))]) 
            / treat[doctor][patient_diseases[i]]
        )
        for i in patients for t in compatible_times[i,doctor]
    )

    m.setObjective(objective_0, gp.GRB.MAXIMIZE)
    m.optimize()

    # Check if the solution is feasible
    # (all patients in schedule are allocated)
    if m.status != gp.GRB.OPTIMAL:
        # print(f"*** doctor {doctor} cannot treat patients {patients}")
        return False, None, None

    # ---------- Doctor-disease rank scores ----------
    # Higher if doctor prefers the disease more
    doctor_num_diseases_can_treat = sum(qualified[doctor])

    doctor_disease_rank_scores = [
        qualified[doctor][k] * (doctor_num_diseases_can_treat - doctor_rank[doctor][k] + 1)
        / doctor_num_diseases_can_treat 
        + (1 - qualified[doctor][k]) * - M1 
        for k in K
    ]

    # Objective 0: Patient satisfaction
    obj0_value = m.ObjVal

    # Objective 1: Total number of appointments
    obj1_value = sum(
        Y[i,doctor,t].x 
        for i in patients for t in compatible_times[i,doctor]
    )
    
    # Objective 2: Doctor satisfaction
    obj2_value = sum(
        (doctor_disease_rank_scores[patient_diseases[i]]) * Y[i,doctor,t].x 
        for i in patients for t in compatible_times[i,doctor]
    )

    Y_values = {
        (i, doctor, t): Y[i, doctor, t].x 
        for (i, doctor, t) in Y 
        if Y[i, doctor, t].x >= 0.9
    }

    return (True, Y_values, (obj0_value, obj1_value, obj2_value))

"""
Find all possible sets of patients that a given doctor can treat

Returns: A set of sets of patients for this doctor, with the scores for each objective
"""
def find_all_patient_sets_for_doctor(doctor: int) -> dict[FrozenSet[int], tuple[tuple[float,float,float], dict[tuple[int,int,int], int]]]:
    patients = patients_doctor_can_treat[doctor]

    schedules_n_patients = dict()
    schedules_n_patients[0] = [([], (0,0,0), {})]
    schedules_n_patients[1] = []
    for patient in patients:
        feasible, Y_values, obj_values = find_best_schedule(doctor, {patient})
        if feasible:
            schedules_n_patients[1].append(([patient], obj_values, Y_values))
    
    n = 2
    while True:
        print(" " * 5, "n =", n)
        schedules_n_patients[n] = []
        for patient_list, _, _ in schedules_n_patients[n - 1]:
            last_patient = patient_list[-1]
            potential_patients = [p for p in patients if p > last_patient]
            for patient in potential_patients:
                new_patient_list = patient_list + [patient]
                feasible, Y_values, obj_values = find_best_schedule(doctor, set(new_patient_list))
                if feasible:
                    schedules_n_patients[n].append((new_patient_list, obj_values, Y_values))
        if not schedules_n_patients[n]:
            break
        n += 1

    all_tuple_schedules = []
    for n in schedules_n_patients:
        all_tuple_schedules.extend(schedules_n_patients[n])

    schedules = {
        frozenset(patient_list): (obj_values, Y_values)
        for patient_list, obj_values, Y_values in all_tuple_schedules
    }
    return schedules

# ---------------- Find all schedules ------------------------
S = dict()
for j in J:
    print("doctor", j)
    S[j] = find_all_patient_sets_for_doctor(j)

# Save schedules to pickle, along with necessary variables to run the huge model and print the schedules
data = {
    "S": S,
    "I": I,
    "J": J,
    "T": T,
    "treat": treat,
    "patient_diseases": patient_diseases,
    "doctor_times": doctor_times
}

with open(f"cg_output.pkl", "wb") as f:
    pickle.dump(data, f)