import gurobipy as gp
# from utils.data_gen import *
# from utils.schedule_printing import *
# from utils.logging_results import *
import pickle
from huge.cg_schedules_timed_multiprocessing import generate_schedules
from huge.cg_schedules_timed import normal_generate_schedules
import os
import time
from utils.data_instance import DataInstance

def get_schedule_data(d:DataInstance, seed, i,j,t,k, gen_use_multiprocessing) -> dict[str, any]:
    multiprocessing_data_name = f"data/cg_subset_output_multiprocessing_seed{seed}_I{i}_J{j}_T{t}_K{k}.pkl"
    normal_data_name = f"data/cg_smart_output_seed{seed}_I{i}_J{j}_T{t}_K{k}.pkl"

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
    if gen_use_multiprocessing:
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
    
    


def make_huge_model(d:DataInstance, seed, i,j,t,k, gen_use_multiprocessing: bool):
    data = get_schedule_data(d, seed, i ,j ,t, k, gen_use_multiprocessing)
    J = data["J"]
    S = data["S"]
    I = data["I"]

    # Initialise model
    m = gp.Model("Huge formulation")
    start_time = time.perf_counter()

    # ============================================================
    # -------------------- Huge formulation ----------------------
    # ============================================================

    m = gp.Model("Doctor scheduling MIP")

    # Doctor schedule
    Z = {
        (j, s): m.addVar(vtype=gp.GRB.BINARY)
        for j in J for s in S[j]
    }

    # Each patient is assigned at most once
    PatientsAreAssignedOnlyOnce = {
        i: m.addConstr(
            gp.quicksum(Z[j, s] for j in J for s in S[j] if i in s) <= 1
        )
        for i in I
    }

    # Each doctor has at most one schedule
    DoctorsHaveOnlyOneSchdeule = {
        j: m.addConstr(
            gp.quicksum(Z[j, s] for s in S[j]) == 1
        )
        for j in J
    }

    objectives = []
    for obj in range(3):
        
        obj_expression = gp.quicksum(S[j][s][0][obj] * Z[j, s] for j in J for s in S[j])
        objectives.append(obj_expression)

        

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