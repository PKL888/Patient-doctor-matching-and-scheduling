import matplotlib.pyplot as plt
import matplotlib.patches as patches

from compact.compatible_times import *
from compact.doctor_available import *
from compact.feasibility import *
from huge.cg_huge import *
from huge.cg_fragments_formulation import *

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 12
})

FEASIBILITY = 1
COMPATIBLE_TIMES = 2
DOCTOR_AVAILABLE = 3
SUBSET_COLUMN_GEN = 8
FRAGMENT_COLUMN_GEN = 9

model_names = {
    FEASIBILITY: "feasibility",
    COMPATIBLE_TIMES: "compatible_times",
    DOCTOR_AVAILABLE: "doctor_available",
    SUBSET_COLUMN_GEN: "subset_column_gen",
    FRAGMENT_COLUMN_GEN: "fragment_column_gen"
}

def left_pad_string(s, length):
    if len(s) >= length:
        return s
    return " " * (length - len(s)) + s 

def expand_schedule(Vs, doctor, d):
    timeline = [-1 for _ in d.T]
    for (i, dd, t) in Vs:
        length = d.treat[doctor][d.patient_diseases[i]]
        for tt in range(t, t + length):
            if tt < len(d.T):
                timeline[tt] = i
    return timeline

def expand_fragments_to_timeline(fragments, doctor, d):
    timeline = [-1 for _ in d.T]
    for f in sorted(fragments, key=lambda fr: fr[START_TIME]):
        for (i, t) in f[PATIENT_TIME_LIST]:
            length = d.treat[doctor][d.patient_diseases[i]]
            for tt in range(t, t + length):
                if tt < len(d.T):
                    timeline[tt] = i
    return timeline

def create_schedule(model_type, d:DataInstance, Y=None, Z=None, W=None, S=None, F=None):
    if model_type == 0:
        schedule = []
        for j in d.J:
            doctor_schedule = [int(patient - 1) for patient in [sum((Y[i,j,tt] * (i + 1)) for k in d.K for i in d.I_k[k] for tt in d.T[max(0, t - d.treat[j][k] + 1):t+1]) for t in d.T]]
            schedule.append(doctor_schedule)
        return schedule
    elif model_type == 1:
        schedule = []
        for j in d.J:
            chosen_s = None
            for s in S[j]:
                if Z[j, s].X > 0.5:
                    chosen_s = s
                    break
            if chosen_s is None:
                doctor_schedule = [-1 for _ in d.T]
            else:
                _, Vs = S[j][chosen_s]
                doctor_schedule = expand_schedule(Vs, j, d)
            schedule.append(doctor_schedule)
        return schedule
    elif model_type == 2:
        schedule = []
        for j in d.J:
            chosen_fragments = []
            for f in F[j]:
                if W[j, f].X > 0.5:
                    chosen_fragments.append(f)

            if len(chosen_fragments) == 0:
                doctor_schedule = [-1 for _ in d.T]
            else:
                doctor_schedule = expand_fragments_to_timeline(chosen_fragments, j, d)
            schedule.append(doctor_schedule)
        return schedule

def print_schedule(model_type, d: DataInstance, schedule):
    padding = len(str(len(d.I)))
    header = "time:     " + " ".join([left_pad_string(str(t), padding) for t in d.T])
    print(header)

    for j, doctor_schedule in zip(d.J, schedule):
        formatted = []
        for t, patient in enumerate(doctor_schedule):
            # Default: show blank if doctor unavailable
            if not d.doctor_times[j][t]:
                s_val = " "
            # Empty slot (available but idle)
            elif patient is None or patient == -1:
                s_val = "-"
            # Treating a patient
            else:
                s_val = str(patient)
            formatted.append(left_pad_string(s_val, padding))

        print(f"doctor {j}:", " ".join(formatted))

def plot_schedule(schedule, model, show_plot, d: DataInstance):
    I = d.I
    J = d.J
    K = d.K
    T = d.T
    w = len(T) / 2 + 1
    h = len(J) / 2 + 1
    fig, ax = plt.subplots(figsize=(w, h))
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
            if d.doctor_times[j][t]:
                # Start of a block
                start = t
                while (t < len(T) and doctor_schedule[t] == patient 
                       and d.doctor_times[j][t]):
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
    
    path = "outputs/images"
    filename = (f"{path}/plot_{model_names[model]}_seed{d.seed}_I{len(I)}_J{len(J)}_K{len(K)}_T{len(T)}.png")
    plt.savefig(filename, dpi=300)
    if show_plot: plt.show()

def get_values_for_model(model, data, d, I=None, J=None, Y=None, Z=None, W=None, S=None, F=None):
    I = I if I is not None else d.I
    J = J if J is not None else d.J
    T = d.T
    if model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]:
        Yvals = {key: Y[key].x for key in Y}
        Ys = {(i, j, t): Yvals.get((i, j, t), 0) for i in I for j in J for t in T}
        
        if model == FEASIBILITY:
            objectives = find_feasibility_objectives(Ys, d, data)
        elif model == COMPATIBLE_TIMES:
            objectives = find_compatible_times_objectives(Ys, data)
        else:
            assert model == DOCTOR_AVAILABLE
            objectives = find_doctor_available_objectives(Ys, data)
        
        return Ys, objectives
    
    elif model == SUBSET_COLUMN_GEN:
        Zvals = {key: Z[key].x for key in Z}
        Zs = {(j, s): Zvals.get((j, s), 0) for j in J for s in S[j]}
        
        objectives = find_huge_objectives(Zs, J, S)

        return Zs, objectives
    
    else:
        assert model == FRAGMENT_COLUMN_GEN
        Ws = {(j, f): W[j, f].x for j in J for f in F[j]}
        
        objectives = find_fragment_objectives(Ws, d, F)

        return Ws, objectives

def print_stats(T, objectives):
    [objective_0, objective_1, objective_2] = objectives

    print("\nStats: ------------------------------------")
    print("Objective 0 (patient satisfaction):  ", round(objective_0, 2))
    print("Objective 1 (num of appointments):   ", round(objective_1))
    print("Objective 2 (doctor satisfaction):   ", round(objective_2, 2))
    print("\nSchedule:", "-" * T * 3)

def optimise_and_print_schedule(model_type, model, m, Y, Z, W, S, F, data, d, show_plot):
    m.optimize()
    
    Vs, objectives = get_values_for_model(model, data, d, Y, Z, W, S, F)
    schedule = create_schedule(model_type, d, Vs, Z, W, S, F)
    
    print_stats(len(d.T), objectives)
    print_schedule(model_type, d, schedule)
    plot_schedule(schedule, model, show_plot, d)