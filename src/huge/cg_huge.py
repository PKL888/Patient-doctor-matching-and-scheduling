import gurobipy as gp
import os
import pickle
import time

from huge.cg_schedules_timed import normal_generate_schedules
from huge.cg_schedules_timed_multiprocessing import generate_schedules
from utils.data_instance import DataInstance

def get_schedule_data(d:DataInstance, seed, i, j, k, t) -> dict[str, any]:
    multiprocessing_data_name = f"data/cg_subset_output_multiprocessing_seed{seed}_I{i}_J{j}_K{k}_T{t}.pkl"
    normal_data_name = f"data/cg_smart_output_seed{seed}_I{i}_J{j}_K{k}_T{t}.pkl"

    if os.path.exists(normal_data_name):
        print(f"Using schedules generated WITHOUT multiprocessing at {normal_data_name}")
        with open(normal_data_name, "rb") as f:
            data = pickle.load(f)
        return data
    
    if os.path.exists(multiprocessing_data_name):
        print(f"Using schedules generated WITH multiprocessing at {multiprocessing_data_name}")
        with open(multiprocessing_data_name, "rb") as f:
            data = pickle.load(f)
        return data
    
    mulit_ans = input(f"Columns have not been generated for seed:{seed} I:{i} J:{j} K:{k} T:{t}. Do you wish to generate columns? (y/n)")
    if not mulit_ans[0] == 'y':
        RuntimeError("Chose not to generate columns")

    use_multi = input("Use multiprocessing to generate columns? (y/n)")[0] == 'y'
    print("Using multiprocessing: ", use_multi)

    base_data_name = f"data/data_seed{seed}_I{i}_J{j}_K{k}_T{t}.pkl"
    if not os.path.exists(base_data_name):
        print(f"no data - {base_data_name} DNE")

    with open(base_data_name, "rb") as f:
        base_data = pickle.load(f)[f"seed_{seed}"]
    print("-"*80)
    print(seed)
    # print(base_data)
    # globals().update(base_data)

    # generate data
    if use_multi:
        generate_schedules(d)
        with open(multiprocessing_data_name, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        normal_generate_schedules(d)
        with open(normal_data_name, "rb") as f:
            data = pickle.load(f)
        return data

    RuntimeError()

def find_huge_objectives(Z, J, S):
    print("\n\n-->", [s for s in S[0]])
    # Objective expressions
    objectives = [
        sum(S[j][s][0][obj] * Z[j, s] for j in J for s in S[j])
        for obj in range(3)
    ]

    return objectives

def make_huge_model(d:DataInstance, seed, i, j, k, t):
    data = get_schedule_data(d, seed, i, j, k, t)
    I = data["I"]
    J = data["J"]
    S = data["S"]

    # Initialise model
    m = gp.Model("Huge formulation")
    start_time = time.perf_counter()

    # Decision variables
    Z = {
        (j, s): m.addVar(vtype=gp.GRB.BINARY)
        for j in J for s in S[j]
    }

    # Constraints
    PatientsAreAssignedOnlyOnce = {
        i: m.addConstr(
            gp.quicksum(Z[j, s] for j in J for s in S[j] if i in s) <= 1
        )
        for i in I
    }

    DoctorsHaveOnlyOneSchdeule = {
        j: m.addConstr(
            gp.quicksum(Z[j, s] for s in S[j]) == 1
        )
        for j in J
    }

    objectives = find_huge_objectives(Z, J, S)

    return m, Z, objectives, start_time - time.perf_counter(), S



    # objectives = []
    # for obj in range(3):
    #     m.setObjective(gp.quicksum(S[j][s][0][obj] * Z[j, s] for j in J for s in S[j]), gp.GRB.MAXIMIZE)

    #     m.setParam("OutputFlag", 0)
    #     m.optimize()

    #     objectives.append(round(m.ObjVal, 2))

    #     print("-" * 50)
    #     print("Maximise objective", obj)
    #     schedule = create_schedule_from_Z(Z, S, J, T, treat, patient_diseases)
    #     print_schedule_from_Z(schedule, I, J, T, doctor_times)

    # print("\n",objectives)