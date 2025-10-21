import gurobipy as gp
import json
import matplotlib.pyplot as plt
from utils.data_gen import *
from src.utils.schedule_printing import *
from src.utils.logging_results import *
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import os

plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern for math
    "font.family": "serif",     # Serif font for text
    "font.size": 12             # Set default font size to 12
})

# ============================================================
# -------------------- Data loading --------------------------
# ============================================================

# Input instance
instance = "data_seed11_I100_J10_K4_T20.pkl"

with open(instance, "rb") as f:
    data = pickle.load(f)

# Expose data fields as globals
globals().update(data)

# Problem dimensions
I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = list(range(problem_size["time periods"]))

# ============================================================
# -------------------- Model Setup ---------------------------
# ============================================================

m = gp.Model("Doctor-Patient Scheduling")

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
        qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1) 
        / doctor_num_diseases_can_treat[j]
        + (1 - qualified[j][k]) * - M1
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
    Y[i, j, t] 
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i, j]
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
        # if not any(all(o >= s for o, s in zip(other, sol)) and any(o > s for o, s in zip(other, sol)) for other in solutions):
        if not (all(o >= s for o, s in zip(other, sol)) for other in solutions):
            non_dominated.append(sol)
    return non_dominated

def pareto_filter_boolean(solutions, current):
    """
    Check if current solution is dominated or not.
    Does not modify the list.
    """
    for other in solutions:
        # if all(o_i >= c_i for o_i, c_i in zip(other, current)) and any(o_i > c_i for o_i, c_i in zip(other, current)):
        if all(o_i >= c_i for o_i, c_i in zip(other, current)):
            return True
    return False

# ============================================================
# ----------------- Pareto computation ----------------------
# ============================================================

def compute_pareto_set(use_slack=True, verbose=True):
    """
    Compute Pareto frontier with optional slack-based dominance tracking.
    Dominated (r, s) pairs are skipped based on previously identified slack regions.
    """

    pareto_solutions = []
    pareto_indices = []
    dominated_solutions = []
    dominated_indices = []

    eps1, r = initial_lower_bound[1], 0

    num_r = int((initial_upper_bound[1] - initial_lower_bound[1]) / delta_eps[1])
    num_s = int((initial_upper_bound[2] - initial_lower_bound[2]) / delta_eps[2])

    while eps1 <= initial_upper_bound[1]:
        if verbose:
            print(f"\n----------- r={r}, eps1={eps1:.3f} -----------")

        EPS1Con.RHS = eps1
        m.update()

        # Maximise objective 2 to find upper bound
        m.setObjective(objective_2)
        EPS2Con.RHS = initial_lower_bound[2]
        obj_stats = tuple(optimise_and_return_stats())
        eps2_upper_bound = obj_stats[2]
        num_s = int((eps2_upper_bound - initial_lower_bound[2]) / delta_eps[2])

        # Now sweep over eps2
        m.setObjective(objective_0)
        eps2, s = initial_lower_bound[2], 0

        while eps2 <= eps2_upper_bound:
            # Skip if this index has been marked dominated
            if (r, s) in dominated_indices:
                if verbose:
                    print(f"----> SKIP (r={r}, s={s})")
                eps2 += delta_eps[2]
                s += 1
                continue

            # Solve model
            EPS2Con.RHS = eps2
            m.update()
            m.Params.OutputFlag = 0
            solution = tuple(optimise_and_return_stats())

            if verbose:
                print(f"SOLVE r={r}, s={s}, eps1={eps1:.2f}, eps2={eps2:.2f}, "
                      f"objs={[round(x,2) for x in solution]}")

            is_dominated = pareto_filter_boolean(pareto_solutions, solution)
            if is_dominated:
                dominated_solutions.append(solution)
                dominated_indices.append((r, s))
            else:
                pareto_solutions.append(solution)
                pareto_indices.append((r, s))

            if use_slack:
                # Identify local dominated area based on slack
                slack2 = max(0, solution[2] - eps2)
                slack1 = max(0, solution[1] - eps1)

                skip_s = int(slack2 / delta_eps[2])
                skip_r = int(slack1 / delta_eps[1])

                if verbose:
                    print(f"      r={r}, s={s}, slack1={slack1:.2f}, slack2={slack2:.2f}, skip_r={skip_r}, skip_s={skip_s}")

                rr_max = min(num_r, r + skip_r)
                ss_max = min(num_s, s + skip_s)

                for ss in range(s+1, ss_max + 1):
                    if (r, ss) not in dominated_indices:
                        dominated_indices.append((r, ss))
                
                for rr in range(r+1, rr_max + 1):
                    if (rr, s) not in dominated_indices:
                        dominated_indices.append((rr, s))
                        
                for rr in range(r+1, rr_max + 1):
                    for ss in range(s+1, ss_max + 1):
                        if (rr, ss) not in dominated_indices:
                            dominated_indices.append((rr, ss))

            eps2 += delta_eps[2]
            s += 1

        eps1 += delta_eps[1]
        r += 1

    # pareto_frontier = pareto_filter(pareto_solutions)
    return pareto_solutions, dominated_solutions, pareto_indices, dominated_indices

# ============================================================
# ------------- Generate or Load Pareto Results --------------
# ============================================================

# Create filename
filename = (
    f"check7_{SEED}"
    f"_I{problem_size['patients']}"
    f"_J{problem_size['doctors']}"
    f"_K{problem_size['diseases']}"
    f"_T{problem_size['time periods']}.pkl"
)

if os.path.exists(filename):
    # Load previously saved results
    with open(filename, "rb") as f:
        output = pickle.load(f)
    print(f"📂 Loaded existing results from {filename}")

    pareto_dense = output.get("pareto_dense")
    dom_dense = output.get("dom_dense")
    pareto_slack = output.get("pareto_slack")
    dom_slack = output.get("dom_slack")

else:
    # Compute both sets if file doesn’t exist
    print("⚙️  Generating new Pareto sets...")
    start_time = time.time()
    print("--- Slack Pareto frontier ---")
    pareto_slack, dom_slack, pareto_ind_slack, dom_ind_slack = compute_pareto_set(use_slack=True, verbose=True)
    slack_time = time.time() - start_time
    print(f"✅ Slack model completed in {slack_time:.2f} seconds")

    print("--- Dense Pareto frontier ---")
    start_time = time.time()
    pareto_dense, dom_dense, pareto_ind_dense, dom_ind_dense = compute_pareto_set(use_slack=False, verbose=True)
    dense_time = time.time() - start_time
    print(f"✅ Dense model completed in {dense_time:.2f} seconds")

    output = {
        "pareto_dense": pareto_dense,
        "dom_dense": dom_dense,
        "pareto_ind_dense": pareto_ind_dense,
        "dom_ind_dense": dom_ind_dense,
        "dense_time": dense_time,
        "pareto_slack": pareto_slack,
        "dom_slack": dom_slack,
        "pareto_ind_slack": pareto_ind_slack,
        "dom_ind_slack": dom_ind_slack,
        "slack_time": slack_time
    }

    with open(filename, "wb") as f:
        pickle.dump(output, f)
    print(f"✅ Saved new results to {filename}")

# ============================================================
# ---------------- Compare Dense vs Slack --------------------
# ============================================================

# pareto_slack, dom_slack, _, _ = compute_pareto_set(use_slack=True,verbose=False)
# pareto_dense, dom_dense, _, _ = compute_pareto_set(use_slack=False,verbose=False)

print(len(pareto_slack))
print(len(pareto_dense))

print()

print(len(dom_slack))
print(len(dom_dense))

print()

print(round(slack_time,2))
print(round(dense_time,2))

# def normalise_pareto(pareto_list, precision=4):
#     """Round and sort Pareto points for stable comparison."""
#     return sorted([tuple(round(x, precision) for x in sol) for sol in pareto_list])

# dense_norm = normalise_pareto(pareto_dense)
# slack_norm = normalise_pareto(pareto_slack)

# if dense_norm == slack_norm:
#     print("✅ Pareto frontiers are identical between dense and slack algorithms.")
# else:
#     print("⚠️ Pareto frontiers differ between algorithms!")
#     print("\n-- Solutions only in dense version --")
#     for sol in set(dense_norm) - set(slack_norm):
#         print(sol)
#     print("\n-- Solutions only in slack version --")
#     for sol in set(slack_norm) - set(dense_norm):
#         print(sol)

# Compute Pareto sets
# pareto_slack = compute_pareto_set(use_slack=True)
# pareto_slack, dom_slack, all_slack = compute_pareto_set(use_slack=True, verbose=True)
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

def plot_pareto_2d_with_dominated(pareto1, dominated_points=None, labels=("Pareto", "Dominated"), save_path=None):
    """Plot 2D projections with optional dominated points."""
    obj0_1, obj1_1, obj2_1 = zip(*pareto1) if pareto1 else ([], [], [])
    d_obj0, d_obj1, d_obj2 = zip(*dominated_points) if dominated_points else ([], [], [])

    cmap = plt.cm.get_cmap("RdYlGn")
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].scatter(obj0_1, obj1_1, c=obj2_1, cmap=cmap, marker="o", label=labels[0])
    if dominated_points: axes[0].scatter(d_obj0, d_obj1, c='grey', s=20, alpha=0.5, marker='x', label=labels[1])
    axes[0].set_xlabel("Objective 0: Patient satisfaction")
    axes[0].set_ylabel("Objective 1: Total appointments")
    axes[0].legend()

    axes[1].scatter(obj0_1, obj2_1, c=obj1_1, cmap=cmap, marker="o", label=labels[0])
    if dominated_points: axes[1].scatter(d_obj0, d_obj2, c='grey', s=20, alpha=0.5, marker='x', label=labels[1])
    axes[1].set_xlabel("Objective 0: Patient satisfaction")
    axes[1].set_ylabel("Objective 2: Doctor satisfaction")
    # axes[1].legend()

    axes[2].scatter(obj1_1, obj2_1, c=obj0_1, cmap=cmap, marker="o", label=labels[0])
    if dominated_points: axes[2].scatter(d_obj1, d_obj2, c='grey', s=20, alpha=0.5, marker='x', label=labels[1])
    axes[2].set_xlabel("Objective 1: Total appointments")
    axes[2].set_ylabel("Objective 2: Doctor satisfaction")
    # axes[2].legend()

    # plt.suptitle("Pareto Comparison" if pareto2 else "Pareto Set")
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.show()

# Example usage: save images
# plot_pareto_comparison_2d(pareto_slack, save_path="pareto_slack_2d.png")
# plot_pareto_2d_with_dominated(pareto_slack, dom_slack, save_path="pareto_slack_2d_with_dominated.png")
# plot_pareto_comparison_2d(pareto_slack, pareto_full, labels=("Slack", "Full"), save_path="pareto_comparison_2d.png")
# plot_pareto_comparison_3d(pareto_slack, save_path="pareto_slack_3d.png")
# plot_pareto_comparison_3d(pareto_slack, pareto_full, labels=("Slack", "Full"), save_path="pareto_comparison_3d.png")