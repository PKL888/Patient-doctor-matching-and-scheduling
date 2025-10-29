import gurobipy as gp
import pickle
import time
from typing import Dict, FrozenSet, Tuple, Optional
from utils.data_instance import DataInstance
import os
from huge.cg_fragments_num_appointments_generation import normal_generate_fragments

NEXT_AVAILABLE_TIME = 2
PATIENT_LIST = 1
START_TIME = 0
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
    
def make_huge_frag_model(d:DataInstance, max_frag_length):
    frag_data = get_fragment_data(d, max_frag_length)
    J = frag_data["J"]
    F = frag_data["F"]
    I = frag_data["I"]
    T = frag_data["T"]
    max_length_fragments_by_next_time = frag_data["max_length_fragments_by_next_time"]
    fragments_by_start_time = frag_data["fragments_by_start_time"]
    
    #     "treat": d.treat,
    #     "patient_diseases": d.patient_diseases,
    #     "doctor_times": d.doctor_times,
    # }

    m_start = time.perf_counter()
    print("Starting to make model")
    m = gp.Model("Fragments")

    # Decision variables
    W = {j: {f: m.addVar(vtype=gp.GRB.BINARY) for f in F[j]} for j in J}
    print("Made W at ", time.perf_counter() - m_start)

    # Constraints
    PatientsAreAssignedOnlyOnce = {
        i: 
        m.addConstr(
            gp.quicksum(W[j][f] for j in J for f in F[j] if i in f[PATIENT_LIST]) <= 1
        )
        for i in I
    }
    print("Assign patients once", time.perf_counter() - m_start)

    DoctorsAreNotOverbooked = {
        (j,t):
        m.addConstr(
            gp.quicksum(W[j][f] for f in W[j] if (f[START_TIME] <= t and t < f[NEXT_AVAILABLE_TIME])) <= 1
        )
        for j in J for t in T
    }
    print("Doctors no overlap", time.perf_counter() - m_start)

    # fragments come after no appointments or a full fragment
    # B is 1 if there was just a break
    B = {(j,t): m.addVar(vtype=gp.GRB.BINARY) for j in J for t in T[1:]}
    for j in J:
        B[j, T[0]] = 1.0
    print("Made Bs", time.perf_counter() - m_start)

    SetBreaks = {
        (j,t):
        m.addConstr(B[j,t] == B[j, t-1] 
            - gp.quicksum(W[j][f] for f in F[j] if f[START_TIME] == t - 1) # start in the previous time period
            + gp.quicksum(W[j][f] for f in F[j] if f[NEXT_AVAILABLE_TIME] == t - 1) # ended in the previous time period
        )
        for j in J for t in T[1:]
    }
    print("Set break flow", time.perf_counter() - m_start)

    SymmetryBreak = {
        (j, t):
        # W[j][f] can only be on if the previous fragment was 3 or it is ont a break
        m.addConstr(
            gp.quicksum(W[j][f] for f in fragments_by_start_time[j][t]) <= 
            # a previous group of 3 appointments
            gp.quicksum(W[j][f] for f in max_length_fragments_by_next_time[j][t])
            + B[j,t]
        )
        for j in J for t in T
    }
    print("Break symmetry using breaks", time.perf_counter() - m_start)

    # -------------------- Objectives ----------------------------
    numberAvailableDoctors = [sum(d.allocate_rank[i][jj] != d.M1 for jj in J) for i in I]
    patientDoctorScore = [[(numberAvailableDoctors[i] - d.allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
    patientTimeScore = [[(d.patient_available[i][1] + 1 - d.patient_time_prefs[i][t]) / d.patient_available[i][1] for t in T] for i in I]
    fragment_patient_scores = {j: {f: 
                                (sum(
                                    patientDoctorScore[i][j] 
                                    + 
                                    sum(patientTimeScore[i][t:min(t + d.treat[j][d.patient_diseases[i]], len(T))]) / d.treat[j][d.patient_diseases[i]]
                                for i,t in f[PATIENT_TIME_LIST])) for f in F[j]} for j in J}


    obj1 = gp.quicksum(W[j][f] * fragment_patient_scores[j][f] for j in J for f in F[j])


    obj0 = gp.quicksum(W[j][f] * len(f[PATIENT_LIST]) for j in J for f in F[j])


    doctor_num_diseases_can_treat = [sum(d.qualified[j]) for j in J]
    doctor_disease_rank_scores = [[d.qualified[j][k] * (doctor_num_diseases_can_treat[j] - d.doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - d.qualified[j][k]) * -d.M1 for k in d.K] for j in J]
    fragment_disease_scores = {j: {f: (sum(doctor_disease_rank_scores[j][d.patient_diseases[p]] for p in f[PATIENT_LIST])) for f in F[j]} for j in J}
    obj2 = gp.quicksum(W[j][f] * fragment_disease_scores[j][f] for j in J for f in F[j])

    objs = [obj0, obj1, obj2]
    return m, W, objs, m_start - time.perf_counter(), F


def find_fragment_objectives(Ws, d: DataInstance, F):
    # print("\n\n-->", [f for f in F[0]])
    # Objective expressions
    # numberAvailableDoctors = [sum(d.allocate_rank[i][jj] != d.M1 for jj in d.J) for i in d.I]
    # patientDoctorScore = [[(d.numberAvailableDoctors[i] - d.allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in d.J] for i in d.I]
    # patientTimeScore = [[(d.patient_available[i][1] + 1 - d.patient_time_prefs[i][t]) / d.patient_available[i][1] for t in d.T] for i in d.I]
    fragment_patient_scores = {j: {f: 
                                (sum(
                                    d.patientDoctorScore[i][j] 
                                    + 
                                    sum(d.patientTimeScore[i][t:min(t + d.treat[j][d.patient_diseases[i]], len(d.T))]) / d.treat[j][d.patient_diseases[i]]
                                for i,t in f[PATIENT_TIME_LIST])) for f in F[j]} for j in d.J}
    obj0 = sum(Ws[j, f] * fragment_patient_scores[j][f] for j in d.J for f in F[j])


    obj1 = sum(Ws[j,f] * len(f[PATIENT_LIST]) for j in d.J for f in F[j])


    # doctor_num_diseases_can_treat = [sum(d.qualified[j]) for j in d.J]
    # doctor_disease_rank_scores = [[d.qualified[j][k] * (doctor_num_diseases_can_treat[j] - d.doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - d.qualified[j][k]) * -d.M1 for k in d.K] for j in d.J]
    fragment_disease_scores = {j: {f: (sum(d.doctor_disease_rank_scores[j][d.patient_diseases[p]] for p in f[PATIENT_LIST])) for f in F[j]} for j in d.J}
    obj2 = sum(Ws[j,f] * fragment_disease_scores[j][f] for j in d.J for f in F[j])

    objs = [obj0, obj1, obj2]

    return objs
