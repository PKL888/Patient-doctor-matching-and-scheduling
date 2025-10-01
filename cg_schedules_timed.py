import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import pickle
import time
from typing import Dict, FrozenSet, Tuple, Optional


file = "data_seed10_I20_J4_K2_T10.pkl"
print("Using", file)
with open(file, "rb") as f:
    data = pickle.load(f)
globals().update(data)

d = gp.Model("dump model")

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

START = 0
DURATION = 1

# ----------------- Timing globals -----------------
time_gen = 0.0
time_mip = 0.0
time_in_mip_solver = 0.0
mip_calls = 0
mip_feasible = 0


def make_small_mip_model_doctor_availability(doctor:int, patients: set[int]):
    m = gp.Model("Small MIP")
    m.setParam("OutputFlag", 0)

    # -------------------- Variables -----------------------------
    Y = {
        (i,doctor,t): m.addVar(vtype=gp.GRB.BINARY)
        for i in patients for t in compatible_times[i,doctor]
    }
    Z = {
        (doctor,t): m.addVar(vtype=gp.GRB.BINARY)
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
            Z[doctor,t] == Z[doctor,t-1]
            + gp.quicksum(
                Y[i,doctor,t-treat[doctor][patient_diseases[i]]]
                for i in patients
                if t-treat[doctor][patient_diseases[i]] in compatible_times[i,doctor]
            )
            - gp.quicksum(
                Y[i,doctor,t]
                for i in patients
                if t in compatible_times[i,doctor]
            )
        )

    m.addConstr(
        Z[doctor,doctor_available[doctor][START]] == 1
        - gp.quicksum(
            Y[i,doctor,doctor_available[doctor][START]]
            for i in patients
            if doctor_available[doctor][START] in compatible_times[i,doctor]
        )
    )

    m.addConstr(
        Z[doctor,doctor_available[doctor][START] + doctor_available[doctor][DURATION]-1]
        + gp.quicksum(
            Y[i,doctor,doctor_available[doctor][START] + doctor_available[doctor][DURATION]
            - treat[doctor][patient_diseases[i]]]
            for i in patients
            if doctor_available[doctor][START] + doctor_available[doctor][DURATION]
            - treat[doctor][patient_diseases[i]] in compatible_times[i,doctor]
        ) == 1
    )

    # -------------------- Objectives ----------------------------
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
        Y[i,doctor,t] * (
            patientDoctorScore[i]
            + sum(patientTimeScore[i][tt] for tt in range(t, min(t + treat[doctor][patient_diseases[i]], len(T))))
            / treat[doctor][patient_diseases[i]]
        )
        for i in patients for t in compatible_times[i,doctor]
    )
    m.setObjective(objective_0, gp.GRB.MAXIMIZE)

    return m, Y


# ==================================================
# Small MIP for a doctor and a set of patients
# ==================================================
def find_best_schedule(doctor: int, patients: set[int]) -> Tuple[bool, Optional[Dict[Tuple[int, int, int], int]], Optional[Tuple[float, float, float]]]:
    global time_mip, mip_calls, mip_feasible, time_in_mip_solver
    mip_calls += 1
    t0 = time.perf_counter()

    m, Y = make_small_mip_model_doctor_availability(doctor, patients)

    t_begin_optimize = time.perf_counter()
    m.optimize()

    end_time = time.perf_counter()
    time_in_mip_solver += end_time - t_begin_optimize
    time_mip += end_time - t0  # accumulate solver time

    if m.status != gp.GRB.OPTIMAL:
        return False, None, None

    mip_feasible += 1

    doctor_num_diseases_can_treat = sum(qualified[doctor])
    doctor_disease_rank_scores = [
        qualified[doctor][k] * (doctor_num_diseases_can_treat - doctor_rank[doctor][k] + 1)
        / doctor_num_diseases_can_treat
        + (1 - qualified[doctor][k]) * - M1
        for k in K
    ]

    obj0_value = m.ObjVal
    obj1_value = sum(Y[i,doctor,t].x for i in patients for t in compatible_times[i,doctor])
    obj2_value = sum(
        (doctor_disease_rank_scores[patient_diseases[i]]) * Y[i,doctor,t].x
        for i in patients for t in compatible_times[i,doctor]
    )

    Y_values = {(i, doctor, t): Y[i, doctor, t].x for (i, doctor, t) in Y if Y[i, doctor, t].x >= 0.9}
    return True, Y_values, (obj0_value, obj1_value, obj2_value)

# ==================================================
# Generate all feasible patient sets for a doctor
# ==================================================
def find_all_patient_sets_for_doctor(doctor: int):
    global time_gen
    t0 = time.perf_counter()

    num_time_periods_available = doctor_available[doctor][1]
    patients = patients_doctor_can_treat[doctor]
    schedules_n_patients = {0: [([], (0,0,0), {}, 0)]}
    schedules_n_patients[1] = []

    for patient in patients:
        feasible, Y_values, obj_values = find_best_schedule(doctor, {patient})
        if feasible:
            #                                                                length of time it takes for doctor to treat that disease
            schedules_n_patients[1].append(([patient], obj_values, Y_values, treat[doctor][patient_diseases[patient]]))

    total_schedules = len(schedules_n_patients[0]) + len(schedules_n_patients[1])

    n = 2
    while True:
        schedules_n_patients[n] = []
        for patient_list, _, _, time_used in schedules_n_patients[n - 1]:
            last_patient = patient_list[-1]
            potential_patients = [(p, treat[doctor][patient_diseases[p]] + time_used) for p in patients if p > last_patient]
            for patient, new_time_used in potential_patients :
                if new_time_used <= num_time_periods_available:
                    new_patient_list = patient_list + [patient]
                    feasible, Y_values, obj_values = find_best_schedule(doctor, set(new_patient_list))
                    if feasible:
                        schedules_n_patients[n].append((new_patient_list, obj_values, Y_values, new_time_used))
                        total_schedules += 1
                        # if not (total_schedules % 10):
                            # print(f"{total_schedules}, {new_patient_list}    ", end = "")
        if not schedules_n_patients[n]:
            break
        n += 1

    time_gen += time.perf_counter() - t0
    print(total_schedules)
    all_tuple_schedules = []
    for n in schedules_n_patients:
        all_tuple_schedules.extend(schedules_n_patients[n])

    return {
        frozenset(patient_list): (obj_values, Y_values)
        for patient_list, obj_values, Y_values, _ in all_tuple_schedules
    }

# ==================================================
# Run across all doctors
# ==================================================
S = {}
for j in J:

    print(f"doctor: {j}, diseases: {diseases_doctor_qualified_for[j]}, treat times: {[treat[j][k] for k in diseases_doctor_qualified_for[j]]}, length available: {doctor_available[j][1]}, ")
    max_appointments = doctor_available[j][1] // min([treat[j][k] for k in diseases_doctor_qualified_for[j]])
    print(f"max appointments: {max_appointments} ", end = "")
    time_before = time.perf_counter()
    S[j] = find_all_patient_sets_for_doctor(j)

    time_taken = time.perf_counter() - time_before
    print(f"time: {time_taken:.2f} s")


print(f"Total wall-clock time in set generation:  {time_gen:.6f} s")
print(f"Total time making and solbing small MIPs: {time_mip:.6f} s")
print(f"Total time solving small MIPs:            {time_in_mip_solver:.6f} s")
print(f"Total MIP calls: {mip_calls}, feasible: {mip_feasible}")

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