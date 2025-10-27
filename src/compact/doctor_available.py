import gurobipy as gp
import time

def find_doctor_available_objectives(Y, data):
    # Objective expressions
    objective_0 = sum(Y[i,j,t] * (patientDoctorScore[i][j] + sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]) for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j])

    objective_1 = sum(Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j])

    objective_2 = sum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j])

    return [objective_0, objective_1, objective_2]

def make_doctor_available_model(data):
    globals().update(data)

    # Initialise model
    m = gp.Model("Doctor availability")
    start_time = time.time()

    # Decision variables
    Y = {
        (i,j,t):
        m.addVar(vtype=gp.GRB.BINARY)
        for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
    }

    Z = {
        (j,t):
        m.addVar(vtype=gp.GRB.BINARY)
        for j in J for t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION]]
    }

    # Constraints
    PatientsAreAssignedOnlyOnce = {
        i:
        m.addConstr(
            gp.quicksum(Y[i,j,t] for j in J_k[k] for t in compatible_times[i,j]) <= 1
        )
        for k in K for i in I_k[k]
    }

    DoctorAvailableConstraint = {
        (j,t):
        m.addConstr(
            Z[j,t] == Z[j,t-1]
                + gp.quicksum(Y[i,j,t-treat[j][k]] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if t-treat[j][k] in compatible_times[i,j])
                - gp.quicksum(Y[i,j,t] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if t in compatible_times[i,j])
        )
        for j in J for t in T[doctor_available[j][START] + 1:doctor_available[j][START] + doctor_available[j][DURATION]]
    }

    DoctorsStartAvailable = {
        j:
        m.addConstr(
            Z[j,doctor_available[j][START]] == 1 
                - gp.quicksum(Y[i,j,doctor_available[j][START]] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if doctor_available[j][START] in compatible_times[i,j])
        )
        for j in J
    }

    DoctorsEndAvailable = {
        j:
        m.addConstr(
            Z[j,doctor_available[j][START] + doctor_available[j][DURATION]-1] 
                + gp.quicksum(Y[i,j,doctor_available[j][START] + doctor_available[j][DURATION]-treat[j][k]] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] in compatible_times[i,j]) == 1
        )
        for j in J
    }

    # Construct model
    m.update()

    # Record before presolve info
    setup_time = time.time() - start_time
    before_presolve_info = {
        "num_variables": m.NumVars,
        "num_constraints": m.NumConstrs,
        "num_nonzeros": m.NumNZs,
        "setup_time_seconds": setup_time
    }
    path = "outputs/logs"
    filename = (f"{path}/presolve_doctor_available_{seed}_I{len(I)}_J{len(J)}_K{len(K)}_T{len(T)}.pkl")
    m.setParam("LogFile", filename)

    objectives = find_doctor_available_objectives(Y, data)

    return m, Y, objectives, setup_time