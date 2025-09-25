import re
import time
from schedule_printing import *

# function for taking from the log and checking against original number of vars and constraints
def parse_presolve_log(m, logfile="gurobi_presolve.log"):
    presolve_info = {
        "rows_removed": 0,
        "columns_removed": 0,
        "nonzeros_removed": 0
    }
    
    with open(logfile, "r") as f:
        for line in f:
            # match "Presolve removed X rows and Y columns"
            match = re.search(r"Presolve removed (\d+) rows? and (\d+) columns?", line)
            if match:
                # calculate the variables and constraints from before presolve minus
                # the rows and cols removed in presolve
                presolve_info["num_variables"] = m.NumVars - int(match.group(2))
                presolve_info["num_constraints"] = m.NumConstrs - int(match.group(1))

    return presolve_info


def optimise_and_print_schedule(m, M1, Y, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs, doctor_times):
    m.optimize()
    Yvals = {key: Y[key].x for key in Y}
    Ys = {(i,j,t): Yvals.get((i,j,t), 0) for i in I for j in J for t in T}

    schedule = create_schedule(Ys, K, J, I_k, T, treat)
    print_stats(Ys, M1, I, J, K, T, I_k, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs)
    print_schedule(schedule, I, J, T, doctor_times)

def optimise_and_collect(objective_name, m, Y, M1, I, J, K, T, I_k, treat, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs):
    start_obj_time = time.time()
    m.optimize()
    end_obj_time = time.time()
    after_presolve_info = parse_presolve_log(m, "gurobi_presolve.log")
    
    after_presolve_info["run_time_seconds"] = end_obj_time - start_obj_time

    Yvals = {key: Y[key].x for key in Y}
    Ys = {(i,j,t): Yvals.get((i,j,t), 0) for i in I for j in J for t in T}

    # Build schedule
    schedule = create_schedule(Ys, K, J, I_k, T, treat)

    # Collect stats
    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [
        [
            qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] 
            + (1 - qualified[j][k]) * -M1 
            for k in K
        ] 
        for j in J
    ]

    stats = {
        "objective": objective_name,
        "objective_value": m.objVal if m.SolCount > 0 else None,
        "runtime": m.Runtime,
        "mip_gap": m.MIPGap if m.IsMIP else None,
        "nodes": m.NodeCount,
        "iterations": m.IterCount,
        "solutions_found": m.SolCount,
        "num_patients_allocated": round(sum(Ys[i,j,t] for i in I for j in J for t in T)),
        "patient_satisfaction": round(sum(
            Ys[i,j,t] * (
                (numberAvailableDoctors[i] - allocate_rank[i][j] + 1)/numberAvailableDoctors[i] 
                + ((patient_available[i][1]) + 1 - patient_time_prefs[i][t])/patient_available[i][1]
            ) 
            for i in I for j in J for t in T)),
        "doctor_satisfaction": round(sum(
            (doctor_disease_rank_scores[j][k]) * Ys[i,j,t] 
            for k in K for i in I_k[k] for j in J for t in T)),
        "appointments_per_doctor": round(sum(Ys[i,j,t] for i in I for j in J for t in T))/len(J),
    }

    # Convert schedule to pickle-friendly structure
    schedule_dict = {f"doctor_{j}": schedule[j] for j in J}

    return {
        "stats": stats,
        "schedule": schedule_dict,
        "after_presolve_info": after_presolve_info
    }