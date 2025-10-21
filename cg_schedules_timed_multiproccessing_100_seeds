import gurobipy as gp
import pickle
import time
from typing import Dict, Tuple, Optional
import multiprocessing

# ==================================================
# Load all seeds
# ==================================================
with open("all_data_100_seeds_I20_J5_K4_T10.pkl", "rb") as f:
    all_data = pickle.load(f)

START_IDX, DURATION_IDX = 0, 1

# ==================================================
# Small MIP for a doctor and a set of patients
# ==================================================
def make_small_mip_model_doctor_availability(
    doctor: int,
    patients: set[int],
    compatible_times,
    doctor_available,
    treat,
    patient_diseases,
    allocate_rank,
    patient_available,
    patient_time_prefs,
    J,
    M1,
):
    m = gp.Model("Small MIP")
    m.setParam("OutputFlag", 0)
    m.setParam("Threads", 1)

    Y = {(i, doctor, t): m.addVar(vtype=gp.GRB.BINARY)
         for i in patients for t in compatible_times[i, doctor]}
    start, duration = doctor_available[doctor]
    Z = {(doctor, t): m.addVar(vtype=gp.GRB.BINARY)
         for t in range(start, start + duration)}

    # Constraints
    for i in patients:
        m.addConstr(gp.quicksum(Y[i, doctor, t] for t in compatible_times[i, doctor]) == 1)

    for t in range(start + 1, start + duration):
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
        Z[doctor, start] == 1
        - gp.quicksum(
            Y[i, doctor, start]
            for i in patients
            if start in compatible_times[i, doctor]
        )
    )

    m.addConstr(
        Z[doctor, start + duration - 1]
        + gp.quicksum(
            Y[i, doctor, start + duration - treat[doctor][patient_diseases[i]]]
            for i in patients
            if start + duration - treat[doctor][patient_diseases[i]] in compatible_times[i, doctor]
        ) == 1
    )

    # Objective
    numberAvailableDoctors = {i: sum(allocate_rank[i][jj] != M1 for jj in J) for i in patients}
    patientDoctorScore = {
        i: ((numberAvailableDoctors[i] - allocate_rank[i][doctor] + 1) / numberAvailableDoctors[i]
            if allocate_rank[i][doctor] != M1 else 0) for i in patients
    }
    patientTimeScore = {
        i: [(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in range(len(patient_time_prefs[i]))]
        for i in patients
    }

    objective_0 = gp.quicksum(
        Y[i, doctor, t] * (
            patientDoctorScore[i]
            + sum(patientTimeScore[i][tt] for tt in range(t, min(t + treat[doctor][patient_diseases[i]], len(patientTimeScore[i]))))
            / treat[doctor][patient_diseases[i]]
        )
        for i in patients for t in compatible_times[i, doctor]
    )
    m.setObjective(objective_0, gp.GRB.MAXIMIZE)

    return m, Y

# ==================================================
# Find best schedule for a doctor & patient set
# ==================================================
def find_best_schedule(
    doctor: int,
    patients: set[int],
    compatible_times,
    doctor_available,
    treat,
    patient_diseases,
    allocate_rank,
    patient_available,
    patient_time_prefs,
    J,
    K,
    qualified,
    doctor_rank,
    M1
):
    m, Y = make_small_mip_model_doctor_availability(
        doctor, patients, compatible_times, doctor_available,
        treat, patient_diseases, allocate_rank, patient_available, patient_time_prefs, J, M1
    )
    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        return False, None, None

    doctor_num_diseases_can_treat = sum(qualified[doctor])
    doctor_disease_rank_scores = [
        qualified[doctor][k] * (doctor_num_diseases_can_treat - doctor_rank[doctor][k] + 1)
        / doctor_num_diseases_can_treat + (1 - qualified[doctor][k]) * -M1
        for k in K
    ]

    obj0_value = m.ObjVal
    obj1_value = sum(Y[i, doctor, t].x for i in patients for t in compatible_times[i, doctor])
    obj2_value = sum(doctor_disease_rank_scores[patient_diseases[i]] * Y[i, doctor, t].x
                     for i in patients for t in compatible_times[i, doctor])

    Y_values = {(i, doctor, t): Y[i, doctor, t].x for (i, doctor, t) in Y if Y[i, doctor, t].x >= 0.9}
    return True, Y_values, (obj0_value, obj1_value, obj2_value)

# ==================================================
# Generate all feasible patient sets for a doctor
# ==================================================
def find_all_patient_sets_for_doctor(
    doctor: int,
    compatible_times,
    doctor_available,
    patients_doctor_can_treat,
    treat,
    patient_diseases,
    allocate_rank,
    patient_available,
    patient_time_prefs,
    J,
    K,
    qualified,
    doctor_rank,
    M1
):
    start, duration = doctor_available[doctor]
    num_time_periods_available = duration
    patients = patients_doctor_can_treat[doctor]
    schedules_n_patients = {0: [([], (0, 0, 0), {}, 0)], 1: []}

    for patient in patients:
        feasible, Y_values, obj_values = find_best_schedule(
            doctor, {patient}, compatible_times, doctor_available,
            treat, patient_diseases, allocate_rank, patient_available,
            patient_time_prefs, J, K, qualified, doctor_rank, M1
        )
        if feasible:
            schedules_n_patients[1].append(([patient], obj_values, Y_values, treat[doctor][patient_diseases[patient]]))

    n = 2
    while True:
        schedules_n_patients[n] = []
        for patient_list, _, _, time_used in schedules_n_patients[n - 1]:
            last_patient = patient_list[-1]
            potential_patients = [(p, treat[doctor][patient_diseases[p]] + time_used) for p in patients if p > last_patient]
            for patient, new_time_used in potential_patients:
                if new_time_used <= num_time_periods_available:
                    new_patient_list = patient_list + [patient]
                    feasible, Y_values, obj_values = find_best_schedule(
                        doctor, set(new_patient_list), compatible_times, doctor_available,
                        treat, patient_diseases, allocate_rank, patient_available,
                        patient_time_prefs, J, K, qualified, doctor_rank, M1
                    )
                    if feasible:
                        schedules_n_patients[n].append((new_patient_list, obj_values, Y_values, new_time_used))
        if not schedules_n_patients[n]:
            break
        n += 1

    all_tuple_schedules = []
    for n in schedules_n_patients:
        all_tuple_schedules.extend(schedules_n_patients[n])

    return {frozenset(patient_list): (obj_values, Y_values) for patient_list, obj_values, Y_values, _ in all_tuple_schedules}

# ==================================================
# Multiprocessing per doctor
# ==================================================
def run_doctor_data(args):
    j, seed_data, K, J, M1 = args
    start_time = time.perf_counter()
    result = find_all_patient_sets_for_doctor(
        j,
        seed_data["compatible_times"],
        seed_data["doctor_available"],
        seed_data["patients_doctor_can_treat"],
        seed_data["treat"],
        seed_data["patient_diseases"],
        seed_data["allocate_rank"],
        seed_data["patient_available"],
        seed_data["patient_time_prefs"],
        J, K,
        seed_data["qualified"],
        seed_data["doctor_rank"],
        M1
    )
    elapsed = time.perf_counter() - start_time
    return j, result, elapsed

# ==================================================
# Run column generation for all seeds
# ==================================================
if __name__ == "__main__":
    default_M1 = 1000
    all_cg_results = {}

    for seed, seed_data in all_data.items():
        print(f"\n===== Seed {seed} =====")
        problem_size = seed_data["problem_size"]
        I = range(problem_size["patients"])
        J = range(problem_size["doctors"])
        K = range(problem_size["diseases"])
        M1 = seed_data.get("M1", default_M1)

        start_seed_timer = time.perf_counter()

        # Prepare multiprocessing arguments
        args_list = [(j, seed_data, K, J, M1) for j in J]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(run_doctor_data, args_list)

        S = {}
        timings = {}
        for j, result, timing in results:
            S[j] = result
            timings[j] = timing

        runtime = time.perf_counter() - start_seed_timer
        #print(f"Seed {seed} finished in {runtime:.2f}s")

        all_cg_results[seed] = {"S": S, "timings": timings, "runtime_seconds": runtime}

    # Save final results
    with open("cg_schedules_timed_multiproccessing_all_100_seed_model_results.pkl", "wb") as f:
        pickle.dump(all_cg_results, f)

    print("\n✅ All seeds completed.")
