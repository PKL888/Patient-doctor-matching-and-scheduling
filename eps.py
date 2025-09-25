import gurobipy as gp
from data_gen import *
import random
import json

instance = "data_seed10_I100_J10_K4_T20.json"

with open(instance, "r") as f:
    data = json.load(f)

# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = [t for t in range(problem_size["time periods"])]

START = 0
DURATION = 1

I_k = [[i for i in I if patient_diseases[i] == k] for k in K]
J_k = [[j for j in J if qualified[j][k]] for k in K]

diseases_doctor_qualified_for = {j: [k for k in K if qualified[j][k]] for j in J}

# diseases_doctor_can_treat_at_time = {(j,t): [k for k in diseases_doctor_qualified_for[j] if t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] + 1]]
#                                      for j in J for t in T}

# patients_with_disease_that_can_be_treated_at_time = \
# {(k,t):
#     [i for i in I_k[k] if t in T[patient_available[i][START]: patient_available[i][START] + patient_available[i][DURATION]]] 
# for k in K for t in T}

compatible_times = {(i,j):
                    T[max(patient_available[i][START], doctor_available[j][START]):
      (max(0, min(patient_available[i][START] + patient_available[i][DURATION], doctor_available[j][START] + doctor_available[j][DURATION]) - treat[j][k] + 1))]
      for k in K for i in I_k[k] for j in J_k[k]}

# Create a binary list for doctor availability per doctor per time period
doctor_times = []
for d in doctor_available:
    time_list = []
    binary_list = []
    for j in range(d[0], d[0]+d[1]):
        time_list.append(j)
    for i in range(len(T)):
        if i in time_list:
            binary_list.append(1)
        else:
            binary_list.append(0)
    doctor_times.append(binary_list)

# Create a binary list for patient availability per patient per time period 
patient_times = []
for p in patient_available:
    time_list = []
    binary_list = []
    for j in range(p[0], p[0]+p[1]):
        time_list.append(j)
    for i in range(len(T)):
        if i in time_list:
            binary_list.append(1)
        else:
            binary_list.append(0)
    patient_times.append(binary_list) 

m = gp.Model("Doctor availability")

##################################################################################
# Variables
Y = {(i,j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
    }

# 1 if doctor is available at time t, 0 otherwise
Z = {(j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for j in J for t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION]]
    }

##################################################################################
# Constraints
PatientsAreAssignedOnlyOnce = \
{(i):
 m.addConstr(gp.quicksum(Y[i,j,t] for j in J_k[k] for t in compatible_times[i,j]) <=1)
for k in K for i in I_k[k]
}


DoctorAvailableConstraint = \
{(j,t):
m.addConstr(Z[j,t] == 
            Z[j,t-1] # previous availability
            + gp.quicksum(Y[i,j,t-treat[j][k]] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if t-treat[j][k] in compatible_times[i,j]) # incoming availability
            - gp.quicksum(Y[i,j,t] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if t in compatible_times[i,j]) # outgoing
            )
for j in J for t in T[doctor_available[j][START] + 1:doctor_available[j][START] + doctor_available[j][DURATION]]}

DoctorsStartAvailable = \
{j:
m.addConstr(Z[j,doctor_available[j][START]] == 1 - gp.quicksum(Y[i,j,doctor_available[j][START]] for k in diseases_doctor_qualified_for[j] for i in I_k[k] if doctor_available[j][START] in compatible_times[i,j]))
for j in J}

DoctorsEndAvailable = \
{j:
m.addConstr(Z[j,doctor_available[j][START] + doctor_available[j][DURATION]-1] + 
            gp.quicksum(Y[i,j,doctor_available[j][START] + doctor_available[j][DURATION]-treat[j][k]] 
                        for k in diseases_doctor_qualified_for[j] 
                        for i in I_k[k] if doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] in compatible_times[i,j]) == 1)
for j in J}

#################################################################
# printing and optimising
def left_pad_string(s, length):
    if len(s) >= length:
        return s
    
    return " " * (length - len(s)) + s 

def create_schedule(Ys):
    schedule = []
    for j in J:
        # print("doctor:", j, "treatment length:", treat[j])
        # print("times available", doctor_times[j])
        # print("start appointment", [sum(Y[i,j,t].x for i in I) for t in T])
        # print("checking ", [sum((Y[i,j,tt].x) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) and not doctor_times[j][t] for t in T])
        doctor_schedule = [int(patient - 1) for patient in [sum((Ys[i,j,tt] * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]]
        # doctor_schedule_with_disease = [(patient, patient_diseases[patient]) for patient in doctor_schedule]
        # print("(patient, disease)", [(int(patient - 1), patient_diseases[int(patient - 1)]) for patient in [sum((Y[i,j,tt].x * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]])
        # print("(patient, disease)", doctor_schedule_with_disease)
        schedule.append(doctor_schedule)
    return schedule

def make_stats(Ys):
    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
    ob1_pat_sat = (sum(Ys[i,j,t] * ((numberAvailableDoctors[i] - allocate_rank[i][j] + 1)/numberAvailableDoctors[i] + ((patient_available[i][1]) + 1 - patient_time_prefs[i][t])/patient_available[i][1]) for i in I for j in J for t in T))
    ob2_total_appts = sum(Ys[i,j,t] for i in I for j in J for t in T)
    ob3_doctor_sat = sum((doctor_disease_rank_scores[j][k]) * Ys[i,j,t] for k in K for i in I_k[k] for j in J for t in T)
    return ob1_pat_sat, ob2_total_appts, ob3_doctor_sat

def print_stats(Ys):
    ob1_pat_sat, ob2_total_appts, ob3_doctor_sat = make_stats(Ys)

    print("Stats: -----------------------------------")
    print("Number of patients allocated:", round(ob2_total_appts))
    print("Patient satisfaction with doctor and time:", round(ob1_pat_sat))
    print("Doctor satisfaction with diseases:", round(ob3_doctor_sat))
    print("Appointments per doctor:", round(sum(Ys[i,j,t] for i in I for j in J for t in T))/len(J))

def print_schedule(schedule):
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))
    for j in J:
        formatted_doctor_schedule = [(patient >= 0) * str(patient) + (patient < 0) * " " + "-" * (1 - doctor_times[j][t]) for t, patient in enumerate(schedule[j])]
        padded_doctor_shedule = [left_pad_string(s, padding) for s in formatted_doctor_schedule]
        print("doctor:", j, " ".join(padded_doctor_shedule))

def optimise_and_print_schedule():
    m.optimize()
    Yvals = {key: Y[key].x for key in Y}
    Ys = {(i,j,t): Yvals.get((i,j,t), 0) for i in I for j in J for t in T}

    schedule = create_schedule(Ys)
    print_stats(Ys)
    print_schedule(schedule)

def optimise_and_return_stats():
    m.optimize()
    Yvals = {key: Y[key].x for key in Y}
    # for all combinations of i j and t, if a match can exist be the Yval, otherwise just 0
    Ys = {(i,j,t): Yvals.get((i,j,t), 0) for i in I for j in J for t in T}
    return make_stats(Ys)

m.setParam("OutputFlag", 0)

# Objective 1: Max. number of matches
print("Objective 1: Max. number of matches")

m.setObjective(gp.quicksum(Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
total_appts_objs = tuple(optimise_and_return_stats())

# Objective 2: Max. patient satisfaction
print("Objective 2: Max. patient satisfaction")

numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]

m.setObjective(gp.quicksum(Y[i,j,t] * 
                           (
                                patientDoctorScore[i][j] 
                                + 
                                sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]
                            )
                           for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
pat_sat_objs = tuple(optimise_and_return_stats())
# m.optimize()

# Objective 3: Max. doctor satisfaction
print("Objective 3: Max. doctor satisfaction")

doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

m.setObjective(gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
doc_sat_objs = tuple(optimise_and_return_stats())

import math

# ---------- Initial bounds and step sizes ----------
# Upper and lower bounds for objectives 1 (appointments) and 2 (doctor satisfaction)
initial_upper_objs_bound = [total_appts_objs[1], doc_sat_objs[2]]
initial_lower_objs_bound = [
    min(pat_sat_objs[1], doc_sat_objs[1]), 
    min(pat_sat_objs[2], total_appts_objs[2])
]

# Step sizes for epsilon increments
deltas = [1, 0.1]  # [delta_eps1, delta_eps2]

# ---------- Objective expressions ----------
# Objective 0: patient satisfaction
objective0_expression = gp.quicksum(
    Y[i,j,t] * (patientDoctorScore[i][j] + 
                sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k])
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
)

# Objective 1: number of appointments
objective1_expression = gp.quicksum(
    Y[i,j,t] 
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
)

# Objective 2: doctor satisfaction
objective2_expression = gp.quicksum(
    (doctor_disease_rank_scores[j][k]) * Y[i,j,t]
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
)

# ---------- Epsilon constraints ----------
EPS1Con = m.addConstr(objective1_expression >= 0)
EPS2Con = m.addConstr(objective2_expression >= 0)

# ---------- Helper: Pareto filter ----------
def pareto_filter(solutions):
    """
    Keep only non-dominated solutions.
    Each solution is a tuple of objective values.
    """
    non_dominated = []
    for sol in solutions:
        dominated = False
        for other in solutions:
            # Check if 'other' (strictly) dominates 'sol'
            if all(o_i >= s_i for o_i, s_i in zip(other, sol)) and any(o_i > s_i for o_i, s_i in zip(other, sol)):
                dominated = True
                break
        if not dominated:
            non_dominated.append(sol)
    return non_dominated

def pareto_filter_boolean(solutions, current):
    """
    Check if current solution is dominated or not.
    """
    dominated = False
    solutions.remove(current)
    for other in solutions:
        if all(o_i >= c_i for o_i, c_i in zip(other, current)) and any(o_i > c_i for o_i, c_i in zip(other, current)):
            dominated = True
    return dominated

# ---------- Pareto solutions set ----------
pareto_solutions = []

# ---------- Outer loop over eps1 ----------
eps1 = initial_lower_objs_bound[0]
r = 0

# Save outputs to list
output = {}

while eps1 <= initial_upper_objs_bound[0]:
    print(f"\n----------- Outer loop r={r} -----------")

    EPS1Con.RHS = eps1
    m.update()

    # ---------- Determine eps2 upper bound ----------
    m.setObjective(objective2_expression)
    EPS2Con.RHS = initial_lower_objs_bound[1]
    obj2_stats = tuple(optimise_and_return_stats())
    obj2_upper_bound = obj2_stats[2]  # maximum achievable objective 2
    m.setObjective(objective0_expression)  # restore primary objective

    # ---------- Inner loop over eps2 ----------
    eps2 = initial_lower_objs_bound[1]
    s = 0

    while eps2 <= obj2_upper_bound:
        EPS2Con.RHS = eps2
        m.update()
        m.Params.OutputFlag = 0

        # Solve with current epsilon constraints
        solution = tuple(optimise_and_return_stats())
        print(f"Inner loop r={r}, s={s}, eps1={round(eps1,3)}, eps2={round(eps2,3)}, objs={[round(x,3) for x in solution]}")

        pareto_solutions.append(solution)

        dominated = pareto_filter_boolean(pareto_solutions, solution)
        output[(r, s, eps1, eps2, solution[0], solution[1], solution[2])] = dominated

        eps2 += deltas[1]
        s += 1

    eps1 += deltas[0]
    r += 1

# ---------- Finished ----------
pareto_frontier = pareto_filter(pareto_solutions)

pareto_sorted = sorted(pareto_frontier, key=lambda sol: (-sol[0], -sol[1], -sol[2]))
all_pareto_sorted = sorted(pareto_solutions, key=lambda sol: (-sol[0], -sol[1], -sol[2]))
dominated_points = [sol for sol in all_pareto_sorted if sol not in pareto_frontier]

print("----- All points found -----")
for sol in all_pareto_sorted:
    print([round(x,3) for x in sol])

print("----- Non-dominated points -----")
for sol in pareto_sorted:
    print([round(x,3) for x in sol])

print("----- Dominated points -----")
for sol in dominated_points:
    print([round(x,3) for x in sol])

print(f"all={len(all_pareto_sorted)}, pareto={len(pareto_sorted)}, dominated={len(dominated_points)}")

import json

with open(f"full_epsilon_output.json", "w") as f:
    json.dump(output, f, indent=4)

# import csv

# with open("solutions_with_status.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Obj0", "Obj1", "Obj2", "Status"])  # headers

#     # Pareto solutions
#     for sol in pareto_sorted:
#         writer.writerow([round(x, 6) for x in sol] + ["Pareto"])

#     # Dominated solutions
#     for sol in dominated_points:
#         writer.writerow([round(x, 6) for x in sol] + ["Dominated"])

# Assume pareto_frontier is a list of tuples (obj1, obj2, obj3)
obj1 = [sol[0] for sol in pareto_frontier]
obj2 = [sol[1] for sol in pareto_frontier]
obj3 = [sol[2] for sol in pareto_frontier]

# --- 2D plots (objective pairs) ---
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_pareto_2d(obj1, obj2, obj3, annotate=False):
    """
    Plot Pareto frontier in 2D projections with colour encoding for the third dimension.

    Parameters
    ----------
    obj1, obj2, obj3 : list or array
        Values of the three objectives for each solution.
    annotate : bool
        If True, adds text labels showing the coloured objective value at each point.
    """
    
    cmap = plt.cm.get_cmap("RdYlGn")  # red=max, green=min
    
    # Normalisation per objective
    norm1 = colors.Normalize(vmin=min(obj3), vmax=max(obj3))
    norm2 = colors.Normalize(vmin=min(obj2), vmax=max(obj2))
    norm3 = colors.Normalize(vmin=min(obj1), vmax=max(obj1))
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Adjust space at bottom for horizontal colourbar
    # plt.subplots_adjust(bottom=0.25)
    
    # --- Obj1 vs Obj2, colour by Obj3 ---
    sc1 = axes[0].scatter(obj1, obj2, c=obj3, cmap=cmap, norm=norm1)
    if annotate:
        for x, y, z in zip(obj1, obj2, obj3):
            axes[0].annotate(str(round(z, 1)), (x, y), fontsize=8,
                             xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel("Objective 1: Matches")
    axes[0].set_ylabel("Objective 2: Patient satisfaction")
    axes[0].xaxis.set_label_position('top')
    axes[0].xaxis.tick_top()
    
    # --- Obj1 vs Obj3, colour by Obj2 ---
    sc2 = axes[1].scatter(obj1, obj3, c=obj2, cmap=cmap, norm=norm2)
    if annotate:
        for x, y, z in zip(obj1, obj2, obj3):
            axes[1].annotate(str(round(y, 1)), (x, z), fontsize=8,
                             xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("Objective 1: Matches")
    axes[1].set_ylabel("Objective 3: Doctor satisfaction")
    axes[1].xaxis.set_label_position('top')
    axes[1].xaxis.tick_top()
    
    # --- Obj2 vs Obj3, colour by Obj1 ---
    sc3 = axes[2].scatter(obj2, obj3, c=obj1, cmap=cmap, norm=norm3)
    if annotate:
        for x, y, z in zip(obj1, obj2, obj3):
            axes[2].annotate(str(round(x, 1)), (y, z), fontsize=8,
                             xytext=(5, 5), textcoords="offset points")
    axes[2].set_xlabel("Objective 2: Patient satisfaction")
    axes[2].set_ylabel("Objective 3: Doctor satisfaction")
    axes[2].xaxis.set_label_position('top')
    axes[2].xaxis.tick_top()
    
    # Horizontal colourbar underneath all plots
    cbar = fig.colorbar(sc3, ax=axes, orientation="horizontal", fraction=0.05)
    cbar.set_label("Objective value (shared scale: green=max, red=min)")
    cbar.ax.set_position([0.1, 0.1, 0.8, 0.03])
    
    plt.show()

# plot_pareto_2d(obj1, obj2, obj3)

from mpl_toolkits.mplot3d import Axes3D

# --- 3D plot ---
def plot_pareto_3d(obj1, obj2, obj3):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.get_cmap("RdYlGn")  # red=max, green=min

    # Use obj3 as colour
    sc = ax.scatter(obj1, obj2, obj3, c=obj3, cmap=cmap, s=50)

    ax.set_xlabel("Objective 1: Patient satisfaction")
    ax.set_ylabel("Objective 2: Appointments")
    ax.set_zlabel("Objective 3: Doctor satisfaction")

    # Add colourbar to show scale
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", label="Doctor satisfaction", fraction=0.05)
    cbar.ax.set_position([0.25, 0.15, 0.6, 0.03])

    plt.show()

# plot_pareto_3d(obj1, obj2, obj3)

def plot_pareto_3d_with_dominated(pareto_points, dominated_points):
    """
    Plot 3D Pareto frontier with dominated points shown distinctly.
    
    Parameters
    ----------
    pareto_points : list of tuples
        [(obj1, obj2, obj3), ...] for non-dominated points.
    dominated_points : list of tuples
        [(obj1, obj2, obj3), ...] for dominated points.
    """
    
    # Split coordinates
    obj1, obj2, obj3 = zip(*pareto_points)
    d_obj1, d_obj2, d_obj3 = zip(*dominated_points) if dominated_points else ([], [], [])
    
    # Normalisation for consistent colour scale
    all_values = list(obj1) + list(obj2) + list(obj3) #+ list(d_obj1) + list(d_obj2) + list(d_obj3)
    norm = colors.Normalize(vmin=min(all_values), vmax=max(all_values))
    cmap = plt.cm.get_cmap("RdYlGn")  # green=min, red=max
    
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pareto frontier points
    sc = ax.scatter(obj1, obj2, obj3, c=obj3, cmap=cmap, norm=norm, s=50, label="Pareto frontier")
    
    # Dominated points
    if dominated_points:
        ax.scatter(d_obj1, d_obj2, d_obj3, c='grey', s=20, alpha=0.5, marker='x', label="Dominated")
    
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    
    # Horizontal colourbar
    cbar = fig.colorbar(sc, orientation='horizontal', fraction=0.03)
    cbar.ax.set_position([0.15, 0.05, 0.7, 0.03])
    cbar.set_label("Objective 3 value (green=min, red=max)")
    
    ax.legend()
    plt.show()

dominated_points = [sol for sol in pareto_solutions if sol not in pareto_frontier]
# plot_pareto_3d_with_dominated(pareto_frontier, dominated_points)
