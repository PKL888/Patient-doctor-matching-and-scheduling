import gurobipy as gp
import time

def find_feasibility_objectives(Y, data):
    globals().update(data)
    # Objective expressions
    objective_0 = sum(Y[i,j,t] * (patientDoctorScore[i][j] + sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k])for k in K for i in I_k[k] for j in J for t in T)

    objective_1 = sum(Y[i,j,t] for i in I for j in J for t in T)
    
    objective_2 = sum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J for t in T)

    return [objective_0, objective_1, objective_2]

def make_feasibility_model(data):
    globals().update(data)

    # Initialise model
    m = gp.Model("Feasibility")
    start_time = time.time()

    # Decision variables
    Y = {
        (i,j,t):
        m.addVar(vtype=gp.GRB.BINARY)
        for i in I for j in J for t in T
    }

    # Constraints
    DoctorsAreNotOverbooked = {
        (j,t):
        m.addConstr(
            gp.quicksum(Y[i,j,tt] for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) <= 1
        )
        for j in J for t in T
    }

    PatientsAreSeenAtMostOnce = {
        i:
        m.addConstr(
            gp.quicksum(Y[i,j,t] for j in J for t in T) <= 1
        )
        for i in I
    }

    FeasibleTime = {
        (j,k,t):
        m.addConstr(
            treat[j][k] * Y[i,j,t] <= sum(doctor_times[j][tt] * patient_times[i][tt] for tt in range(t, min(t + treat[j][k], len(T))))
        )
        for k in K for i in I_k[k] for j in J for t in T
    }

    DoctorsQualified = {
        (i,j,k,t):
        m.addConstr( 
            Y[i,j,t] <= qualified[j][k]
        )
        for k in K for i in I_k[k] for j in J for t in T
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
    filename = (f"{path}/presolve_feasibility_{seed}_I{len(I)}_J{len(J)}_K{len(K)}_T{len(T)}.pkl")
    m.setParam("LogFile", filename)

    objectives = find_feasibility_objectives(Y, data)
    
    return m, Y, objectives, setup_time