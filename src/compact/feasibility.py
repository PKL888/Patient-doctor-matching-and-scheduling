import gurobipy as gp
import time

def make_feasibility_model(data):
    globals().update(data)

    m = gp.Model("Doctor patient feasibility")
    start_time = time.time()

    # Variables
    Y = {(i,j,t):
        m.addVar(vtype=gp.GRB.BINARY)
        for i in I for j in J for t in T}

    # Constraints
    DoctorsAreNotOverbooked = \
    {(j,t):
    m.addConstr(gp.quicksum(Y[i,j,tt] for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) <= 1)
    for j in J for t in T}

    PatientsAreSeenAtMostOnce = \
    {i:
    m.addConstr(gp.quicksum(Y[i,j,t] for j in J for t in T) <= 1)
    for i in I}

    FeasibleTime = \
    {(j,k,t):
    m.addConstr(treat[j][k] * Y[i,j,t] <= sum(doctor_times[j][tt] * patient_times[i][tt] for tt in range(t, min(t + treat[j][k], len(T)))))
    for k in K for i in I_k[k] for j in J for t in T}

    DoctorsQualified = \
    {(i,j,k,t):
    m.addConstr( Y[i,j,t] <= qualified[j][k])
    for k in K for i in I_k[k] for j in J for t in T}

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



    objective_0 = gp.quicksum(Y[i,j,t] for i in I for j in J for t in T)

    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
    patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]

    objective_1 = gp.quicksum(Y[i,j,t] * (patientDoctorScore[i][j] + 
                                           sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / 
                                           treat[j][k])
                               for k in K for i in I_k[k] for j in J for t in T)
    
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

    objective_2 = gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J for t in T)

    
    return m, Y, [objective_0, objective_1, objective_2], setup_time
    