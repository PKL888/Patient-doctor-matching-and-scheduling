import gurobipy as gp
import multiprocessing
import pickle
import time
from typing import Dict, Tuple, Optional

from compact.compatible_times import *
from compact.doctor_available import *
from compact.feasibility import *
from utils.data_instance import DataInstance
from utils.logging_results import *

START = 0
DURATION = 1

# ----------------- Timing globals -----------------
time_gen = 0.0
time_mip = 0.0
time_in_mip_solver = 0.0
mip_calls = 0
mip_feasible = 0

def make_small_mip_model_doctor_availability(doctor:int, patients: set[int], d: DataInstance):
    m = gp.Model("Small MIP")
    m.setParam("OutputFlag", 0)
    m.setParam("Threads", 1)

    T = d.T
    J = d.J
    # I = d.I
    # K = d.K


    # -------------------- Variables -----------------------------
    Y = {
        (i,doctor,t): m.addVar(vtype=gp.GRB.BINARY)
        for i in patients for t in d.compatible_times[i,doctor]
    }
    Z = {
        (doctor,t): m.addVar(vtype=gp.GRB.BINARY)
        for t in T[
            d.doctor_available[doctor][START]:
            d.doctor_available[doctor][START] + d.doctor_available[doctor][DURATION]
        ]
    }

    # -------------------- Constraints ---------------------------
    for i in patients:
        m.addConstr(gp.quicksum(Y[i, doctor, t] for t in d.compatible_times[i, doctor]) == 1)

    for t in T[
        d.doctor_available[doctor][START] + 1:
        d.doctor_available[doctor][START] + d.doctor_available[doctor][DURATION]
    ]:
        m.addConstr(
            Z[doctor,t] == Z[doctor,t-1]
            + gp.quicksum(
                Y[i,doctor,t-d.treat[doctor][d.patient_diseases[i]]]
                for i in patients
                if t-d.treat[doctor][d.patient_diseases[i]] in d.compatible_times[i,doctor]
            )
            - gp.quicksum(
                Y[i,doctor,t]
                for i in patients
                if t in d.compatible_times[i,doctor]
            )
        )

    m.addConstr(
        Z[doctor,d.doctor_available[doctor][START]] == 1
        - gp.quicksum(
            Y[i,doctor,d.doctor_available[doctor][START]]
            for i in patients
            if d.doctor_available[doctor][START] in d.compatible_times[i,doctor]
        )
    )

    m.addConstr(
        Z[doctor,d.doctor_available[doctor][START] + d.doctor_available[doctor][DURATION]-1]
        + gp.quicksum(
            Y[i,doctor,d.doctor_available[doctor][START] + d.doctor_available[doctor][DURATION]
            - d.treat[doctor][d.patient_diseases[i]]]
            for i in patients
            if d.doctor_available[doctor][START] + d.doctor_available[doctor][DURATION]
            - d.treat[doctor][d.patient_diseases[i]] in d.compatible_times[i,doctor]
        ) == 1
    )

    # -------------------- Objectives ----------------------------
    numberAvailableDoctors = {
        i: sum(d.allocate_rank[i][jj] != d.M1 for jj in J)
        for i in patients
    }
    patientDoctorScore = {
        i: ((numberAvailableDoctors[i] - d.allocate_rank[i][doctor] + 1) / numberAvailableDoctors[i]
            if d.allocate_rank[i][doctor] != d.M1 else 0)
        for i in patients
    }
    patientTimeScore = {
        i: [(d.patient_available[i][1] + 1 - d.patient_time_prefs[i][t]) / d.patient_available[i][1] for t in T]
        for i in patients
    }

    objective_0 = gp.quicksum(
        Y[i,doctor,t] * (
            patientDoctorScore[i]
            + sum(patientTimeScore[i][tt] for tt in range(t, min(t + d.treat[doctor][d.patient_diseases[i]], len(T))))
            / d.treat[doctor][d.patient_diseases[i]]
        )
        for i in patients for t in d.compatible_times[i,doctor]
    )
    m.setObjective(objective_0, gp.GRB.MAXIMIZE)

    return m, Y


# ==================================================
# Small MIP for a doctor and a set of patients
# ==================================================
def find_best_schedule(model, data, d: DataInstance, I, J) -> Tuple[bool, Optional[Dict[Tuple[int, int, int], int]], Optional[Tuple[float, float, float]]]:
    global time_mip, mip_calls, mip_feasible, time_in_mip_solver
    mip_calls += 1
    t0 = time.perf_counter()

    m, Y, _, _ = make_doctor_available_model(d)

    doctor = J
    patients = set(I)

    t_begin_optimize = time.perf_counter()
    m.optimize()

    end_time = time.perf_counter()
    time_in_mip_solver += end_time - t_begin_optimize
    time_mip += end_time - t0

    if m.status != gp.GRB.OPTIMAL:
        return False, None, None

    mip_feasible += 1

    Ys, objectives = get_values_for_model(model, data, d, I, J, Y=Y)

    return True, Ys, objectives

def find_all_patient_sets_for_doctor(model, doctor: int, data, d: DataInstance):
    global time_gen
    t0 = time.perf_counter()

    num_time_periods_available = d.doctor_available[doctor][1]
    patients = d.patients_doctor_can_treat[doctor]
    schedules_n_patients = {0: [([], (0.0,0.0,0.0), {}, 0)]}
    schedules_n_patients[1] = []

    J = doctor
    for patient in patients:
        I = {patient}
        feasible, Y_values, obj_values = find_best_schedule(model, data, d, I, J)
        if feasible:
            schedules_n_patients[1].append(([patient], obj_values, Y_values, d.treat[doctor][d.patient_diseases[patient]]))

    total_schedules = len(schedules_n_patients[0]) + len(schedules_n_patients[1])

    n = 2
    while True:
        schedules_n_patients[n] = []
        for patient_list, _, _, time_used in schedules_n_patients[n - 1]:
            last_patient = patient_list[-1]
            potential_patients = [(p, d.treat[doctor][d.patient_diseases[p]] + time_used) for p in patients if p > last_patient]
            for patient, new_time_used in potential_patients :
                if new_time_used <= num_time_periods_available:
                    new_patient_list = patient_list + [patient]
                    feasible, Y_values, obj_values = find_best_schedule(model, data, d)
                    if feasible:
                        schedules_n_patients[n].append((new_patient_list, obj_values, Y_values, new_time_used))
                        total_schedules += 1
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
    
def run_doctor_data(model, j: int, data, d: DataInstance):
    print(f"doctor: {j}, diseases: {d.diseases_doctor_qualified_for[j]}, treat times: {[d.treat[j][k] for k in d.diseases_doctor_qualified_for[j]]}, length available: {d.doctor_available[j][1]}, ")
    max_appointments = d.doctor_available[j][1] // min([d.treat[j][k] for k in d.diseases_doctor_qualified_for[j]])
    print(f"doctor: {j}, max appointments: {max_appointments} ", end = "")
    time_before = time.perf_counter()
    result = find_all_patient_sets_for_doctor(model, j, data, d)

    time_taken = time.perf_counter() - time_before

    print(f"doctor: {j}, time: {time_taken:.2f} s")

    return j, result, time_taken

def run_doctor_data_wrapper(args):
    model, j, data, d = args
    return run_doctor_data(model, j, data, d)

def generate_schedules(model, data, d: DataInstance):
    start_general_timer = time.perf_counter()

    manager = multiprocessing.Manager()
    S = manager.dict()
    timings = manager.dict()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_doctor_data_wrapper, [(model, j, data, d) for j in d.J])

    S = {}
    timings = {}
    for j, result, timing in results:
        S[j] = result
        timings[j] = timing

    end_general_timer = time.perf_counter()
    print(f"Total wall-clock time in parrallel:  {end_general_timer - start_general_timer:.6f} s")

    print("Per-doctor process times:")
    for j in sorted(timings):
        print(f"  Doctor {j}: {timings[j]:.2f} s")

    # print(f"Total wall-clock time in set generation:  {time_gen:.6f} s")
    # print(f"Total time making and solving small MIPs: {time_mip.value:.6f} s")
    # print(f"Total time solving small MIPs:            {time_in_mip_solver:.6f} s")
    # print(f"Total MIP calls (of which feasible):      {mip_calls.value} ({mip_feasible.value})")
    data = {
        "S": S,
        "I": d.I,
        "J": d.J,
        "T": d.T,
        "treat": d.treat,
        "patient_diseases": d.patient_diseases,
        "doctor_times": d.doctor_times
    }

    with open(f"data/cg_subset_output_multiprocessing_seed{d.seed}_I{len(d.I)}_J{len(d.J)}_K{len(d.K)}_T{len(d.T)}.pkl", "wb") as f:
        pickle.dump(data, f)
    return S