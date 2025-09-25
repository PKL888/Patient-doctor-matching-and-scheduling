import gurobipy as gp
import json
import matplotlib.pyplot as plt
from data_gen import *
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# -------------------- Data loading --------------------------
# ============================================================

# Input instance
instance = "data_seed10_I100_J10_K4_T20.json"

with open(instance, "r") as f:
    data = json.load(f)

# Expose data fields as globals
globals().update(data)

# Problem dimensions
I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = list(range(problem_size["time periods"]))

START, DURATION = 0, 1 # tuple index shorthand

# Pre-computed sets
I_k = [[i for i in I if patient_diseases[i] == k] for k in K]
J_k = [[j for j in J if qualified[j][k]] for k in K]

# Map: each doctor → diseases they can treat
diseases_doctor_qualified_for = {j: [k for k in K if qualified[j][k]] for j in J}

# Map: compatible times (patient i, doctor j) → feasible time slots
compatible_times = {(i, j): T[
    max(patient_available[i][START], doctor_available[j][START]) :
    max(0, min(
            patient_available[i][START] + patient_available[i][DURATION],
            doctor_available[j][START] + doctor_available[j][DURATION],
        ) - treat[j][k] + 1, )] 
    for k in K for i in I_k[k] for j in J_k[k]
}

# ============================================================
# ----------- Availability encoding (binary lists) -----------
# ============================================================

def build_availability_binaries(availabilities, total_time):
    """
    Convert availability intervals into binary lists over time.

    Parameters
    ----------
    availabilities : list of (start, duration)
        Availability intervals for each individual (doctor or patient).
    total_time : int
        Number of time periods.

    Returns
    -------
    list of lists
        Binary availability vectors, one per individual.
    """
    binaries = []
    for start, dur in availabilities:
        active_times = set(range(start, start + dur))
        binaries.append([1 if t in active_times else 0 for t in range(total_time)])
    return binaries

# Binary availability vectors
doctor_times = build_availability_binaries(doctor_available, len(T))
patient_times = build_availability_binaries(patient_available, len(T))

# ============================================================
# -------------------- Model Setup ---------------------------
# ============================================================

m = gp.Model("Doctor–Patient Scheduling")

# -------------------- Variables -----------------------------

# Y[i,j,t] = 1 if patient i is assigned to doctor j starting at time t
Y = {
    (i, j, t): m.addVar(vtype=gp.GRB.BINARY)
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]
}

# Z[j,t] = 1 if doctor j is available at time t
Z = {
    (j, t): m.addVar(vtype=gp.GRB.BINARY)
    for j in J for t in T[
        doctor_available[j][START] :
        doctor_available[j][START] + doctor_available[j][DURATION]
    ]
}

# -------------------- Constraints ---------------------------

# Each patient is assigned at most once
PatientsAreAssignedOnlyOnce = {
    i: m.addConstr(
        gp.quicksum(
            Y[i, j, t]
            for j in J_k[k] for t in compatible_times[i, j]
        ) <= 1
    )
    for k in K for i in I_k[k]
}

# Doctor availability dynamics
DoctorAvailableConstraint = {
    (j, t): m.addConstr(Z[j, t] == Z[j, t - 1]
        + gp.quicksum(
            Y[i, j, t - treat[j][k]]
            for k in diseases_doctor_qualified_for[j] for i in I_k[k] 
            if t - treat[j][k] in compatible_times[i, j]
        )
        - gp.quicksum(
            Y[i, j, t]
            for k in diseases_doctor_qualified_for[j] for i in I_k[k]
            if t in compatible_times[i, j]
        )
    )
    for j in J for t in T[
        doctor_available[j][START] + 1 :
        doctor_available[j][START] + doctor_available[j][DURATION]
    ]
}

# Initial doctor availability
DoctorsStartAvailable = {
    j: m.addConstr(Z[j, doctor_available[j][START]] == 1
        - gp.quicksum(
            Y[i, j, doctor_available[j][START]]
            for k in diseases_doctor_qualified_for[j] for i in I_k[k]
            if doctor_available[j][START] in compatible_times[i, j]
        )
    )
    for j in J
}

# Final doctor availability
DoctorsEndAvailable = {
    j: m.addConstr(
        Z[j, doctor_available[j][START] + doctor_available[j][DURATION] - 1]
        + gp.quicksum(
            Y[i, j, doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k]]
            for k in diseases_doctor_qualified_for[j]
            for i in I_k[k]
            if doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] in compatible_times[i, j]
        )
        == 1
    )
    for j in J
}

# ============================================================
# ----------------- Helper Functions -------------------------
# ============================================================

def left_pad_string(s: str, length: int) -> str:
    """Pad string `s` with spaces on the left to a fixed length."""
    return s if len(s) >= length else " " * (length - len(s)) + s

def create_schedule(Ys: dict) -> list:
    """
    Build doctor schedules from solution values of Y.
    
    Each schedule[j] is a list of patient indices (-1 if no patient) for each time t.
    """
    schedule = []
    for j in J:
        doctor_schedule = [
            int(sum(
                Ys[i, j, tt] * (i + 1)  # +1 to avoid zero, then subtract later if needed
                for k in K for i in I_k[k]
                for tt in T[max(0, t - treat[j][k] + 1): t + 1]
            ) - 1)  # convert back to 0-indexed patient
            for t in T
        ]
        schedule.append(doctor_schedule)
    return schedule

def make_stats(Ys: dict) -> tuple:
    """
    Compute key statistics from a solution Ys:
    1. Patient satisfaction
    2. Total number of appointments
    3. Doctor satisfaction
    """
    # Patient satisfaction
    ob1_pat_sat = sum(
        Ys[i, j, t] * (
            patientDoctorScore[i][j] +
            patientTimeScore[i][t]
        )
        for i in I for j in J for t in T
    )

    # Total number of appointments
    ob2_total_appts = sum(Ys[i, j, t] for i in I for j in J for t in T)

    # Doctor satisfaction
    ob3_doctor_sat = sum(
        doctor_disease_rank_scores[j][k] * Ys[i, j, t]
        for k in K for i in I_k[k] for j in J for t in T
    )

    return ob1_pat_sat, ob2_total_appts, ob3_doctor_sat

def print_stats(Ys: dict):
    """Print summary statistics for a given solution Ys."""
    ob1, ob2, ob3 = make_stats(Ys)
    total_appointments_per_doctor = ob2 / len(J)

    print("Stats: -----------------------------------")
    print("Number of patients allocated:", round(ob2))
    print("Patient satisfaction (doctor + time):", round(ob1))
    print("Doctor satisfaction:", round(ob3))
    print("Average appointments per doctor:", round(total_appointments_per_doctor))

def print_schedule(schedule: list):
    """Print doctor schedules in a readable time vs doctor format."""
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))

    for j, doc_schedule in enumerate(schedule):
        formatted_schedule = [
            str(patient) if patient >= 0 else " "
            for t, patient in enumerate(doc_schedule)
        ]
        # Add dash if doctor not available at time t
        formatted_schedule = [
            f"{s}-" if doctor_times[j][t] == 0 else s
            for t, s in enumerate(formatted_schedule)
        ]
        padded_schedule = [left_pad_string(s, padding) for s in formatted_schedule]
        print("doctor:", j, " ".join(padded_schedule))

def optimise_and_return_Ys() -> dict:
    """
    Optimise the model and return Ys dict for all (i,j,t), filling 0 if no appointment.
    """
    m.optimize()
    Yvals = {key: Y[key].x for key in Y}
    Ys = {(i, j, t): Yvals.get((i, j, t), 0) for i in I for j in J for t in T}
    return Ys

def optimise_and_print_schedule():
    """
    Optimise the model and print the resulting schedule and statistics.
    """
    Ys = optimise_and_return_Ys()
    schedule = create_schedule(Ys)
    print_stats(Ys)
    print_schedule(schedule)

def optimise_and_return_stats() -> tuple:
    """
    Optimise the model and return statistics (patient sat, total appointments, doctor sat)
    """
    Ys = optimise_and_return_Ys()
    return make_stats(Ys)

# ============================================================
# ----------------- Precompute score lists -------------------
# ============================================================

# ---------- Number of doctors available to each patient ----------
numberAvailableDoctors = [sum(allocate_rank[i][j] != M1 for j in J) for i in I]

# ---------- Patient-doctor score ----------
# Higher if patient prefers the doctor more
patientDoctorScore = [
    [(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J]
    for i in I
]

# ---------- Patient-time score ----------
# Higher if patient prefers the time more
patientTimeScore = [
    [(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T]
    for i in I
]

# ---------- Doctor-disease rank scores ----------
# Higher if doctor prefers the disease more
doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
doctor_disease_rank_scores = [
    [
        qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1) / doctor_num_diseases_can_treat[j]
        + (1 - qualified[j][k]) * -M1
        for k in K
    ]
    for j in J
]

# ============================================================
# ----------------- Objective expressions -------------------
# ============================================================

# Objective 0: Patient satisfaction
objective_0 = gp.quicksum(
    Y[i, j, t] * (
        patientDoctorScore[i][j] +
        sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]
    )
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]
)

# Objective 1: Total number of appointments
objective_1 = gp.quicksum(
    Y[i, j, t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]
)

# Objective 2: Doctor satisfaction
objective_2 = gp.quicksum(
    doctor_disease_rank_scores[j][k] * Y[i, j, t]
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]
)

# ============================================================
# ----------------- Initial optimisation --------------------
# ============================================================

m.setParam("OutputFlag", 0)  # silent mode

# Maximise each objective individually to get bounds
m.setObjective(objective_0, gp.GRB.MAXIMIZE)
pat_sat_objs = optimise_and_return_stats()

m.setObjective(objective_1, gp.GRB.MAXIMIZE)
total_appts_objs = optimise_and_return_stats()

m.setObjective(objective_2, gp.GRB.MAXIMIZE)
doc_sat_objs = optimise_and_return_stats()

# ============================================================
# ----------------- Epsilon-constraint setup ----------------
# ============================================================

initial_upper_bound = [None, total_appts_objs[1], doc_sat_objs[2]]
initial_lower_bound = [
    None,
    min(pat_sat_objs[1], doc_sat_objs[1]), 
    min(pat_sat_objs[2], total_appts_objs[2])
]

delta_eps = [None, 1, 0.1]  # [delta_eps0, delta_eps1, delta_eps2]

EPS1Con = m.addConstr(objective_1 >= 0)
EPS2Con = m.addConstr(objective_2 >= 0)

# ============================================================
# ----------------- Pareto filtering ------------------------
# ============================================================

def pareto_filter(solutions):
    """Keep only non-dominated solutions."""
    non_dominated = []
    for sol in solutions:
        if not any(all(o >= s for o, s in zip(other, sol)) and any(o > s for o, s in zip(other, sol)) for other in solutions):
            non_dominated.append(sol)
    return non_dominated

def pareto_filter_boolean(solutions, current):
    """
    Check if current solution is dominated or not.
    Does not modify the list.
    """
    for other in solutions:
        if all(o_i >= c_i for o_i, c_i in zip(other, current)) and any(o_i > c_i for o_i, c_i in zip(other, current)):
            return True
    return False

# ============================================================
# ----------------- Pareto computation ----------------------
# ============================================================

def compute_pareto_set(use_slack=True, verbose=True):
    """
    Compute Pareto frontier with optional slack algorithm.

    Objective numbering:
    0: Patient satisfaction
    1: Total appointments
    2: Doctor satisfaction
    """
    # pareto_solutions, all_solutions, output_status = [], [], {}
    pareto_solutions = []

    eps1, r = initial_lower_bound[1], 0

    if use_slack:
        num_s = int((initial_upper_bound[2] - initial_lower_bound[2]) / delta_eps[2]) + 1
        num_passes_1 = [0] * num_s
        bound = 0

    while eps1 <= initial_upper_bound[1]:
        if verbose:
            print(f"\n----------- r={r}, eps1={eps1} -----------")

        EPS1Con.RHS = eps1
        m.update()

        m.setObjective(objective_2)
        EPS2Con.RHS = initial_lower_bound[2]
        obj_stats = tuple(optimise_and_return_stats())
        eps2_upper_bound = obj_stats[2]
        m.setObjective(objective_0)

        eps2, s = initial_lower_bound[2], 0
        while eps2 <= eps2_upper_bound:
            if use_slack and num_passes_1[s] > bound and bound > 0:
                if verbose:
                    print(f"----> SKIP-eps1 r={r}, s={s}, eps1={eps1:.3f}, eps2={eps2:.3f} "
                          f"(num_passes_1[{s}]={num_passes_1[s]}, bound={bound})")
                num_passes_1[s] -= 1
                eps2 += delta_eps[2]
                s += 1
                continue

            EPS2Con.RHS = eps2
            m.update()
            m.Params.OutputFlag = 0

            solution = tuple(optimise_and_return_stats())
            # all_solutions.append(solution)

            if verbose:
                print(f"  SOLVE r={r}, s={s}, eps1={eps1:.3f}, eps2={eps2:.3f}, "
                      f"objs={[round(x,2) for x in solution]}")

            dominated = pareto_filter_boolean(pareto_solutions, solution)
            # output_status[(r, s, eps1, eps2, solution)] = dominated
            if not dominated:
                pareto_solutions.append(solution)

            if use_slack:
                slack2 = solution[2] - eps2
                num_passes_2 = max(1, int(slack2 / delta_eps[2]) + 1)

                slack1 = solution[1] - eps1
                num_skips_1 = max(1, int(slack1 / delta_eps[1]) + 1)

                num_passes_1[s] = max(num_passes_1[s], num_skips_1)
                for ss in range(s, min(s + num_passes_2, num_s)):
                    num_passes_1[ss] = max(num_passes_1[ss], num_skips_1)

                # for step in range(1, num_passes_2):
                    # if verbose:
                    #     print(f"----> SKIP-eps2 r={r}, s={s+step}, eps1={eps1:.3f}, "
                    #           f"eps2={eps2 + step*delta_eps[2]:.3f} "
                    #           f"(jumped {num_passes_2} steps)")

                eps2 += num_passes_2 * delta_eps[2]
                s += num_passes_2
            else:
                eps2 += delta_eps[2]
                s += 1

        if use_slack:
            bound += 1

        eps1 += delta_eps[1]
        r += 1
    
    pareto_frontier = pareto_filter(pareto_solutions)
    # dominated_points = [sol for sol in all_solutions if sol not in pareto_frontier]
    return pareto_frontier
    # return pareto_frontier, dominated_points, all_solutions, output_status

# Compute Pareto sets
pareto_slack = compute_pareto_set(use_slack=True)
# pareto_slack, dom_slack, all_slack, status_slack = compute_pareto_set(use_slack=True)
# pareto_full, dom_full, all_full, status_full = compute_pareto_set(use_slack=False)

# import json

# slack_data = {
#     "pareto_slack": pareto_slack,
#     "dominated_slack": dom_slack,
#     "all_solutions_slack": all_slack,
#     # "status_slack": status_slack
# }

# full_data = {
#     "pareto_full": pareto_full,
#     "dominated_full": dom_full,
#     "all_solutions_full": all_full,
#     # "status_full": status_full
# }

# with open("slack_output.json", "w") as f:
#     json.dump(slack_data, f, indent=4)

# with open("full_output.json", "w") as f:
#     json.dump(full_data, f, indent=4)

# ============================================================
# ----------------- Pareto plotting -------------------------
# ============================================================

def plot_pareto_comparison_2d(pareto1, pareto2=None, labels=("Pareto 1", "Pareto 2"), save_path=None):
    """Plot 2D projections with optional comparison."""
    obj0_1, obj1_1, obj2_1 = zip(*pareto1) if pareto1 else ([], [], [])
    if pareto2: obj0_2, obj1_2, obj2_2 = zip(*pareto2) if pareto2 else ([], [], [])

    cmap = plt.cm.get_cmap("RdYlGn")
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].scatter(obj0_1, obj1_1, c=obj2_1, cmap=cmap, marker="o", label=labels[0])
    if pareto2: axes[0].scatter(obj0_2, obj1_2, c=obj2_2, cmap=cmap, marker="x", label=labels[1])
    axes[0].set_xlabel("Objective 0: Patient satisfaction")
    axes[0].set_ylabel("Objective 1: Matches")
    axes[0].legend()

    axes[1].scatter(obj0_1, obj2_1, c=obj1_1, cmap=cmap, marker="o", label=labels[0])
    if pareto2: axes[1].scatter(obj0_2, obj2_2, c=obj1_2, cmap=cmap, marker="x", label=labels[1])
    axes[1].set_xlabel("Objective 0: Patient satisfaction")
    axes[1].set_ylabel("Objective 2: Doctor satisfaction")
    axes[1].legend()

    axes[2].scatter(obj1_1, obj2_1, c=obj0_1, cmap=cmap, marker="o", label=labels[0])
    if pareto2: axes[2].scatter(obj1_2, obj2_2, c=obj0_2, cmap=cmap, marker="x", label=labels[1])
    axes[2].set_xlabel("Objective 1: Matches")
    axes[2].set_ylabel("Objective 2: Doctor satisfaction")
    axes[2].legend()

    plt.suptitle("Pareto Comparison" if pareto2 else "Pareto Set")
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.show()

def plot_pareto_comparison_3d(pareto1, pareto2=None, labels=("Pareto 1", "Pareto 2"), save_path=None):
    """Plot 3D Pareto sets with optional comparison."""
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap("RdYlGn")

    obj0_1, obj1_1, obj2_1 = zip(*pareto1) if pareto1 else ([], [], [])
    ax.scatter(obj0_1, obj1_1, obj2_1, cmap=cmap, s=50, label=labels[0])

    if pareto2:
        obj0_2, obj1_2, obj2_2 = zip(*pareto2) if pareto2 else ([], [], [])
        ax.scatter(obj0_2, obj1_2, obj2_2, c='red', s=50, label=labels[1])

    ax.set_xlabel("Objective 0: Patient satisfaction")
    ax.set_ylabel("Objective 1: Matches")
    ax.set_zlabel("Objective 2: Doctor satisfaction")
    ax.legend()
    plt.title("Pareto Comparison" if pareto2 else "Pareto Set")
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.show()

# Example usage: save images
plot_pareto_comparison_2d(pareto_slack, save_path="pareto_slack_2d.png")
# plot_pareto_comparison_2d(pareto_slack, pareto_full, labels=("Slack", "Full"), save_path="pareto_comparison_2d.png")
plot_pareto_comparison_3d(pareto_slack, save_path="pareto_slack_3d.png")
# plot_pareto_comparison_3d(pareto_slack, pareto_full, labels=("Slack", "Full"), save_path="pareto_comparison_3d.png")