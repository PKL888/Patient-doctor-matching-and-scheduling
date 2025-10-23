import re
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 12
})

def left_pad_string(s, length):
    if len(s) >= length:
        return s
    return " " * (length - len(s)) + s 

def create_schedule(Ys, K, J, I_k, T, treat):
    schedule = []
    for j in J:
        doctor_schedule = [int(patient - 1) for patient in [sum((Ys[i,j,tt] * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]]
        schedule.append(doctor_schedule)
    return schedule

def print_stats(Ys, M1, I, J, K, T, I_k, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs):
    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
    print("Stats: -----------------------------------")
    print("Number of patients allocated:", round(sum(Ys[i,j,t] for i in I for j in J for t in T)))
    print("Patient satisfaction with doctor and time:", round(sum(Ys[i,j,t] * ((numberAvailableDoctors[i] - allocate_rank[i][j] + 1)/numberAvailableDoctors[i] + ((patient_available[i][1]) + 1 - patient_time_prefs[i][t])/patient_available[i][1]) for i in I for j in J for t in T)))
    print("Doctor satisfaction with diseases:", round(sum((doctor_disease_rank_scores[j][k]) * Ys[i,j,t] for k in K for i in I_k[k] for j in J for t in T)))
    print("Appointments per doctor:", round(sum(Ys[i,j,t] for i in I for j in J for t in T))/len(J))

def print_schedule(schedule, I, J, T, doctor_times):
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))
    for j in J:
        formatted_doctor_schedule = []
        for t, patient in enumerate(schedule[j]):
            if patient is None or patient < 0:  # nobody scheduled
                if doctor_times[j][t]:  
                    s_val = "-"   # available but not treating
                else:
                    s_val = " "   # doctor unavailable
            else:
                s_val = str(patient)  # treating patient
            formatted_doctor_schedule.append(left_pad_string(s_val, padding))
        print("doctor:", j, " ".join(formatted_doctor_schedule))

def expand_schedule(Y_values, doctor, T, treat, patient_diseases):
    timeline = [-1 for _ in T]
    for (i, d, t) in Y_values:
        length = treat[doctor][patient_diseases[i]]
        for tt in range(t, t + length):
            if tt < len(T):
                timeline[tt] = i
    return timeline

def create_schedule_from_Z(Z, S, J, T, treat, patient_diseases):
    schedule = []
    for j in J:
        chosen_s = None
        for s in S[j]:
            if Z[j, s].X > 0.5:
                chosen_s = s
                break
        if chosen_s is None:
            doctor_schedule = [-1 for _ in T]
        else:
            _, Y_values = S[j][chosen_s]
            doctor_schedule = expand_schedule(Y_values, j, T, treat, patient_diseases)
        schedule.append(doctor_schedule)
    return schedule

def print_schedule_from_Z(schedule, I, J, T, doctor_times):
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))
    for j, doctor_schedule in zip(J, schedule):
        formatted = []
        for t, patient in enumerate(doctor_schedule):
            if patient == -1:
                s_val = "-"
            else:
                s_val = str(patient)
            if not doctor_times[j][t]:
                s_val = " "
            formatted.append(left_pad_string(s_val, padding))
        print("doctor:", j, " ".join(formatted))

def plot_schedule(schedule, I, J, T, doctor_times, path):
    fig, ax = plt.subplots(figsize=(8, 2))

    cmap = plt.cm.gist_rainbow  # rainbow colormap
    n_patients = len(I)

    # Assign a unique colour for each patient
    patient_colors = {patient: cmap(i / n_patients) for i, patient in enumerate(I)}

    # Iterate over doctors (rows)
    for row, j in enumerate(J):
        doctor_schedule = schedule[row]

        t = 0
        while t < len(T):
            patient = doctor_schedule[t]
            if doctor_times[j][t]:
                # Start of a block
                start = t
                while (t < len(T) and doctor_schedule[t] == patient 
                       and doctor_times[j][t]):
                    t += 1
                end = t

                # Draw rectangle for this patient
                height = 0.85
                if patient != -1:
                    ax.add_patch(
                        patches.Rectangle(
                            (start, len(J) - row - 1 - height/2),
                            end - start,
                            height,
                            facecolor=patient_colors[patient], 
                            alpha=0.6
                        )
                    )
                    # Add label I#
                    ax.text(
                        (start + end) / 2,
                        len(J) - row - 1,
                        fr"$I_{{{patient}}}$",
                        ha="center", va="center", weight="bold"
                    )
                else:
                    ax.add_patch(
                        patches.Rectangle(
                            (start, len(J) - row - 1 - height/2),
                            end - start,
                            height,
                            facecolor="gray", 
                            alpha=0.6
                        )
                    )
            else:
                t += 1

    # Set y-ticks as doctor labels
    ax.set_yticks(range(len(J)))
    ax.set_yticklabels([fr"$J_{{{j}}}$" for j in J[::-1]])  # reverse order for top-down
    ax.set_xticks(T)
    ax.set_xlabel("Time periods")
    ax.set_ylabel("Doctors")
    # ax.set_title("Schedules", weight="bold")

    ax.set_xlim(0, len(T))
    ax.set_ylim(-0.5, len(J) - 0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()


# function for taking from the log and checking against original number of vars and constraints
def parse_presolve_log(m, logfile="outputs/logs/gurobi_presolve.log"):
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
    plot_schedule(schedule, I, J, T, doctor_times, path="plot.png")

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