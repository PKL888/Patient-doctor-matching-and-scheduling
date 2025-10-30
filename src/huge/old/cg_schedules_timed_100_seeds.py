import gurobipy as gp
import pickle
import time
from typing import Dict, FrozenSet, Tuple, Optional
from data_gen import *
from schedule_printing import *
from logging_results import *

# ==================================================
# Load all data (many seeds)
# ==================================================
with open("all_data_100_seeds_I20_J5_K4_T10.pkl", "rb") as f:
    all_data = pickle.load(f)

# ==================================================
# Define helper functions (from your column generation code)
# ==================================================

def make_small_mip_model_doctor_availability(doctor: int, patients: set[int]):
    m = gp.Model("Small MIP")
    m.setParam("OutputFlag", 0)

    Y = {
        (i, doctor, t): m.addVar(vtype=gp.GRB.BINARY)
        for i in patients for t in compatible_times[i, doctor]
    }
    Z = {
        (doctor, t): m.addVar(vtype=gp.GRB.BINARY)
        for t in T[
            doctor_available[doctor][START]:
            doctor_available[doctor][START] + doctor_available[doctor][DURATION]
        ]
    }

    # -------------------- Constraints ---------------------------
    for i in patients:
        m.addConstr(gp.quicksum(Y[i, doctor, t] for t in compatible_times[i, doctor]) == 1)

    for t in T[
        doctor_available[doctor][START] + 1:
        doctor_available[doctor][START] + doctor_available[doctor][DURATION]
    ]:
        m.addConstr(
            Z[doctor, t] == Z[doctor, t - 1]
            + gp.quicksum(
                Y[i, doctor, t - treat[doctor][patient_diseases[i]]]
                for i in patients
                if t - treat[doctor][patient_diseases[i]] in compatible_times[i, doctor]
            )
            - gp.quicksum(
                Y[i, doctor, t]
                for i in patients
                if t in compatible_times[i, doctor]
            )
        )

    m.addConstr(
        Z[doctor, doctor_available[doctor][START]] == 1
        - gp.quicksum(
            Y[i, doctor, doctor_available[doctor][START]]
            for i in patients
            if doctor_available[doctor][START] in compatible_times[i, doctor]
        )
    )

    m.addConstr(
        Z[doctor, doctor_available[doctor][START] + doctor_available[doctor][DURATION] - 1]
        + gp.quicksum(
            Y[i, doctor, doctor_available[doctor][START] + doctor_available[doctor][DURATION]
            - treat[doctor][patient_diseases[i]]]
            for i in patients
            if doctor_available[doctor][START] + doctor_available[doctor][DURATION]
            - treat[doctor][patient_diseases[i]] in compatible_times[i, doctor]
        ) == 1
    )

    # -------------------- Objective ----------------------------
    numberAvailableDoctors = {
        i: sum(allocate_rank[i][jj] != M1 for jj in J)
        for i in patients
    }
    patientDoctorScore = {
        i: ((numberAvailableDoctors[i] - allocate_rank[i][doctor] + 1) / numberAvailableDoctors[i]
            if allocate_rank[i][doctor] != M1 else 0)
        for i in patients
    }
    patientTimeScore = {
        i: [(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T]
        for i in patients
    }

    objective_0 = gp.quicksum(
        Y[i, doctor, t] * (
            patientDoctorScore[i]
            + sum(patientTimeScore[i][tt] for tt in range(t, min(t + treat[doctor][patient_diseases[i]], len(T))))
            / treat[doctor][patient_diseases[i]]
        )
        for i in patients for t in compatible_times[i, doctor]
    )
    m.setObjective(objective_0, gp.GRB.MAXIMIZE)
    return m, Y


def find_best_schedule(doctor: int, patients: set[int]):
    m, Y = make_small_mip_model_doctor_availability(doctor, patients)
    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        return False, None, None

    doctor_num_diseases_can_treat = sum(qualified[doctor])
    doctor_disease_rank_scores = [
        qualified[doctor][k] * (doctor_num_diseases_can_treat - doctor_rank[doctor][k] + 1)
        / doctor_num_diseases_can_treat
        + (1 - qualified[doctor][k]) * -M1
        for k in K
    ]

    obj0_value = m.ObjVal
    obj1_value = sum(Y[i, doctor, t].x for i in patients for t in compatible_times[i, doctor])
    obj2_value = sum(
        (doctor_disease_rank_scores[patient_diseases[i]]) * Y[i, doctor, t].x
        for i in patients for t in compatible_times[i, doctor]
    )
    Y_values = {(i, doctor, t): Y[i, doctor, t].x for (i, doctor, t) in Y if Y[i, doctor, t].x >= 0.9}
    return True, Y_values, (obj0_value, obj1_value, obj2_value)


def find_all_patient_sets_for_doctor(doctor: int):
    num_time_periods_available = doctor_available[doctor][1]
    patients = patients_doctor_can_treat[doctor]
    schedules_n_patients = {0: [([], (0, 0, 0), {}, 0)]}
    schedules_n_patients[1] = []

    for patient in patients:
        feasible, Y_values, obj_values = find_best_schedule(doctor, {patient})
        if feasible:
            schedules_n_patients[1].append(([patient], obj_values, Y_values, treat[doctor][patient_diseases[patient]]))

    total_schedules = len(schedules_n_patients[0]) + len(schedules_n_patients[1])

    n = 2
    while True:
        schedules_n_patients[n] = []
        for patient_list, _, _, time_used in schedules_n_patients[n - 1]:
            last_patient = patient_list[-1]
            potential_patients = [(p, treat[doctor][patient_diseases[p]] + time_used) for p in patients if p > last_patient]
            for patient, new_time_used in potential_patients:
                if new_time_used <= num_time_periods_available:
                    new_patient_list = patient_list + [patient]
                    feasible, Y_values, obj_values = find_best_schedule(doctor, set(new_patient_list))
                    if feasible:
                        schedules_n_patients[n].append((new_patient_list, obj_values, Y_values, new_time_used))
                        total_schedules += 1
        if not schedules_n_patients[n]:
            break
        n += 1

    all_tuple_schedules = []
    for n in schedules_n_patients:
        all_tuple_schedules.extend(schedules_n_patients[n])

    return {
        frozenset(patient_list): (obj_values, Y_values)
        for patient_list, obj_values, Y_values, _ in all_tuple_schedules
    }

# ==================================================
# RUN FOR ALL SEEDS (just like your first script)
# ==================================================
all_cg_results = {}

for seed, data in all_data.items():
    print(f"\n========== Running Column Generation for Seed {seed} ==========")
    globals().update(data)

    I = range(problem_size["patients"])
    J = range(problem_size["doctors"])
    K = range(problem_size["diseases"])
    T = range(problem_size["time periods"])

    START, DURATION = 0, 1

    S = {}
    start_time = time.time()

    for j in J:
        #print(f"Doctor {j} - diseases: {diseases_doctor_qualified_for[j]} - length available: {doctor_available[j][DURATION]}")
        S[j] = find_all_patient_sets_for_doctor(j)

    runtime = time.time() - start_time
    all_cg_results[seed] = {
        "S": S,
        "runtime_seconds": runtime
    }

# final save
with open("cg_schedules_timed_all_100_seeds_model_results.pkl", "wb") as f:
    pickle.dump(all_cg_results, f)

print("\n✅ All seeds completed and saved.")
