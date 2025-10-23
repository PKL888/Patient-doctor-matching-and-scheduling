import gurobipy as gp
import time
import pickle
from src.utils.data_gen import *
from src.utils.schedule_printing import *
from src.utils.logging_results import *

with open("all_data_1000_seeds_I100_J10_K4_T20.pkl", "rb") as f:
    all_data = pickle.load(f)

all_model_results = {}
objectives = ["max_matches", "patient_satisfaction", "doctor_satisfaction"]
all_model_results = {obj: {} for obj in objectives}

def make_compatible_times_model(data):
    globals().update(data)

    # Initialise model
    m = gp.Model("Compatible times")
    start_time = time.time()

    # Decision variables
    Y = {
        (i, j, t):
        m.addVar(vtype=gp.GRB.BINARY)
        for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]
    }

    # Constraints
    DoctorsAreNotOverbooked = {
        (j, t):
        m.addConstr(
            gp.quicksum(
                Y[i, j, tt] 
                for k in K if j in J_k[k] 
                for i in I_k[k] 
                for tt in T[max(0, t - treat[j][k] + 1):t + 1] 
                if tt in compatible_times[i, j]
            ) <= 1
        )
        for j in J for t in T
    }

    PatientsAreAssignedOnlyOnce = {
        i:
        m.addConstr(
            gp.quicksum(Y[i, j, t] for j in J_k[k] for t in compatible_times[i, j]) <= 1
        )
        for k in K for i in I_k[k]
    }

    # Construct model
    m.update()
    setup_time = time.time() - start_time
    before_presolve_info = {
        "num_variables": m.NumVars,
        "num_constraints": m.NumConstrs,
        "num_nonzeros": m.NumNZs,
        "setup_time_seconds": setup_time
    }
    m.setParam("LogFile", "gurobi_presolve.log")

    # m.setParam("OutputFlag", 0)

    # Objective expressions
    objective_0 = m.setObjective(
        gp.quicksum(
            Y[i, j, t] * (
                patientDoctorScore[i][j] +
                sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]
            )
            for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]
        ),
        gp.GRB.MAXIMIZE
    )

    objective_1 = m.setObjective(
        gp.quicksum(Y[i, j, t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]),
        gp.GRB.MAXIMIZE
    )

    objective_2 = m.setObjective(
        gp.quicksum(doctor_disease_rank_scores[j][k] * Y[i, j, t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]),
        gp.GRB.MAXIMIZE
    )

    # Solve and collect results
    # model_results = optimise_and_collect(objective, m, Y, M1, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs)

    # return {"before_presolve_info": before_presolve_info, "model_results": model_results}
    return m, Y, [objective_0, objective_1, objective_2]
