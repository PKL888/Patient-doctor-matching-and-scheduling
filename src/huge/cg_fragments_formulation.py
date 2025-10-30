import gurobipy as gp
import os
import pickle
import time

from huge.cg_fragments_num_appointments_generation import normal_generate_fragments
from utils.data_instance import DataInstance

START_TIME = 0
PATIENT_LIST = 1
NEXT_AVAILABLE_TIME = 2
PATIENT_TIME_LIST = 3

def get_fragment_data(d:DataInstance, max_frag_length: int) -> dict[str, any]:
    fragment_data_name = f"data/cg_fragments_maxfraglength{max_frag_length}_seed{d.seed}_I{len(d.I)}_J{len(d.J)}_T{len(d.T)}_K{len(d.K)}.pkl"
    
    if os.path.exists(fragment_data_name):
        print(f"Found fragment data at {fragment_data_name}")

        with open(fragment_data_name, "rb") as f:
            data = pickle.load(f)
        return data
    
    new_fragment_data = normal_generate_fragments(d, max_frag_length)
    return new_fragment_data

def find_fragment_objectives(W, d: DataInstance, F):
    J = d.J

    fragment_patient_scores = {
        j: {
            f: (
                sum(d.patientDoctorScore[i][j] + sum(d.patientTimeScore[i][t:min(t + d.treat[j][d.patient_diseases[i]], len(d.T))]) / d.treat[j][d.patient_diseases[i]] 
                    for i,t in f[PATIENT_TIME_LIST]
                )
            ) for f in F[j]
        } for j in J
    }

    fragment_disease_scores = {
        j: {
            f: (
            sum(d.doctor_disease_rank_scores[j][d.patient_diseases[p]] 
                for p in f[PATIENT_LIST])
            ) for f in F[j]
        } for j in J
    }

    # Objective expressions
    obj0 = sum(W[j, f] * fragment_patient_scores[j][f] for j in J for f in F[j])
    obj1 = sum(W[j, f] * len(f[PATIENT_LIST]) for j in J for f in F[j])
    obj2 = sum(W[j, f] * fragment_disease_scores[j][f] for j in J for f in F[j])

    objectives = [obj0, obj1, obj2]

    return objectives
 
def make_huge_frag_model(d:DataInstance, max_frag_length):
    frag_data = get_fragment_data(d, max_frag_length)
    J = frag_data["J"]
    F = frag_data["F"]
    I = frag_data["I"]
    T = frag_data["T"]
    print("\nFragments:")
    for f, value in F.items():
        print(f, value)
    max_length_fragments_by_next_time = frag_data["max_length_fragments_by_next_time"]
    fragments_by_start_time = frag_data["fragments_by_start_time"]

    m_start = time.perf_counter()
    print("Starting to make fragments model")
    m = gp.Model("Fragments")

    # Decision variables
    W = {(j, f): m.addVar(vtype=gp.GRB.BINARY) for j in J for f in F[j]}
    print("Created W variables:", round(time.perf_counter() - m_start, 2), "seconds")

    # Constraints
    PatientsAreAssignedOnlyOnce = {
        i: 
        m.addConstr(
            gp.quicksum(W[j, f] for j in J for f in F[j] if i in f[PATIENT_LIST]) <= 1
        )
        for i in I
    }
    print("Assigned patients once:", round(time.perf_counter() - m_start, 2), "seconds")

    DoctorsAreNotOverbooked = {
        (j,t):
        m.addConstr(
            gp.quicksum(W[j, f] for f in F[j] if f[START_TIME] <= t < f[NEXT_AVAILABLE_TIME]) <= 1
        )
        for j in J for t in T
    }
    print("Ensured doctors do not overlap:", round(time.perf_counter() - m_start, 2), "seconds")

    # Fragments come after no appointments or a full fragment
    # B is 1 if there was just a break
    B = {(j,t): m.addVar(vtype=gp.GRB.BINARY) for j in J for t in T[1:]}
    for j in J:
        B[j, T[0]] = 1.0
    print("Created B variables:", round(time.perf_counter() - m_start, 2), "seconds")

    SetBreaks = {
        (j,t):
        m.addConstr(B[j,t] == B[j, t-1] 
            - gp.quicksum(W[j, f] for f in F[j] if f[START_TIME] == t - 1) # start in the previous time period
            + gp.quicksum(W[j, f] for f in F[j] if f[NEXT_AVAILABLE_TIME] == t - 1) # ended in the previous time period
        )
        for j in J for t in T[1:]
    }
    print("Ensured set breaks:", round(time.perf_counter() - m_start, 2), "seconds")

    SymmetryBreak = {
        (j, t):
        # W[j, f] can only be on if the previous fragment was max length or it is ont a break
        m.addConstr(
            gp.quicksum(W[j, f] for f in fragments_by_start_time[j][t]) <= 
            # A previous group of max length appointments
            gp.quicksum(W[j, f] for f in max_length_fragments_by_next_time[j][t])
            + B[j,t]
        )
        for j in J for t in T
    }
    print("Broke symmetry:", round(time.perf_counter() - m_start, 2), "seconds")

    objectives = find_fragment_objectives(W, d, F)

    return m, W, objectives, m_start - time.perf_counter(), F