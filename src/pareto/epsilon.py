import gurobipy as gp
from math import ceil
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.compact.compatible_times import *
from src.compact.doctor_available import *
from src.compact.feasibility import *
from src.utils.data_gen import *
from src.utils.logging_results import *

FEASIBILITY = 1
COMPATIBLE_TIMES = 2
DOCTOR_AVAILABLE = 3
SUBSET_COLUMN_GEN = 8
FRAGMENT_COLUMN_GEN = 9

def pareto_filter_boolean(solutions, current):
    for other in solutions:
        if all(o_i >= c_i - 1e-5 for o_i, c_i in zip(other, current)):
            return True
    return False

def mini_opti(m, Y, data, model):
    globals().update(data)

    m.optimize()

    Yvals = {key: Y[key].x for key in Y}
    Ys = {(i, j, t): Yvals.get((i, j, t), 0) for i in I for j in J for t in T}

    if model == FEASIBILITY:
        obj_stats = find_feasibility_objectives(Ys, data)
    elif model == COMPATIBLE_TIMES:
        obj_stats = find_compatible_times_objectives(Ys, data)
    elif model == DOCTOR_AVAILABLE:
        obj_stats = find_doctor_available_objectives(Ys, data)
    else:
        print("Model does not exist in Pareto options...")
        return None

    return obj_stats

def compute_pareto_set(m, Y, objectives, data, model,
                       initial_lower_bound, initial_upper_bound,
                       EPS1Con, EPS2Con, delta_eps,
                       use_slack=True, verbose=True):
    """
    Compute Pareto frontier with optional slack-based dominance tracking.
    Dominated (r, s) pairs are skipped based on previously identified slack regions.
    """

    [objective_0, objective_1, objective_2] = objectives

    pareto_solutions = []
    pareto_indices = []
    dominated_solutions = []
    dominated_indices = []

    eps1, r = initial_lower_bound[1], 0

    num_r = ceil((initial_upper_bound[1] - initial_lower_bound[1]) / delta_eps[1])
    num_s = ceil((initial_upper_bound[2] - initial_lower_bound[2]) / delta_eps[2])

    while eps1 <= initial_upper_bound[1]:
        if verbose:
            print(f"\n---------------------- {r=}, {eps1=:.3f} ----------------------\n")

        EPS1Con.RHS = eps1
        m.update()

        # Maximise objective 2 to find upper bound
        m.setObjective(objective_2)
        EPS2Con.RHS = initial_lower_bound[2]
        obj_stats = mini_opti(m, Y, data, model)
        eps2_upper_bound = obj_stats[2]
        num_s = ceil((eps2_upper_bound - initial_lower_bound[2]) / delta_eps[2])

        # Now sweep over eps2
        m.setObjective(objective_0)
        eps2, s = initial_lower_bound[2], 0

        while eps2 <= eps2_upper_bound:
            # Skip if this index has been marked dominated
            if (r, s) in dominated_indices:
                if verbose:
                    print(f"----> SKIP ({r=}, {s=})")
                eps2 += delta_eps[2]
                s += 1
                continue

            # Solve model
            EPS2Con.RHS = eps2
            m.update()
            m.Params.OutputFlag = 0
            solution = mini_opti(m, Y, data, model)

            if verbose:
                print(f"SOLVE {r=}, {s=}, {eps1=:.2f}, {eps2=:.2f}, "
                      f"objs={[round(x,2) for x in solution]}")

            is_dominated = pareto_filter_boolean(pareto_solutions, solution)
            if is_dominated:
                dominated_solutions.append(solution)
                dominated_indices.append((r, s))
            else:
                pareto_solutions.append(solution)
                pareto_indices.append((r, s))

            if use_slack:
                slack2 = solution[2] - eps2
                slack1 = solution[1] - eps1

                skip_s = int(slack2 / delta_eps[2])
                skip_r = int(slack1 / delta_eps[1])

                if verbose:
                    print(f"      {r=}, {s=}, {slack1=:.2f}, {slack2=:.2f}, {skip_r=}, {skip_s=}")

                rr_max = min(num_r, r + skip_r)
                ss_max = min(num_s, s + skip_s)

                # Iterate over dominated region
                for rr in range(r, rr_max + 1):
                    for ss in range(s, ss_max + 1):
                        if (rr, ss) not in dominated_indices:
                            dominated_indices.append((rr, ss))

            eps2 += delta_eps[2]
            s += 1

        eps1 += delta_eps[1]
        r += 1

    return pareto_solutions, dominated_solutions, pareto_indices, dominated_indices

def make_pareto_frontier(data, m, Y, objectives, model, dense = True):
    # Initial optimisation: maximise each objective individually to get bounds
    [objective_0, objective_1, objective_2] = objectives

    m.setObjective(objective_0, gp.GRB.MAXIMIZE)
    pat_sat_objs = mini_opti(m, Y, data, model)

    m.setObjective(objective_1, gp.GRB.MAXIMIZE)
    total_appts_objs = mini_opti(m, Y, data, model)

    m.setObjective(objective_2, gp.GRB.MAXIMIZE)
    doc_sat_objs = mini_opti(m, Y, data, model)

    # Epsilon-constraint setup
    initial_upper_bound = [None, total_appts_objs[1], doc_sat_objs[2]]
    initial_lower_bound = [
        None,
        min(pat_sat_objs[1], doc_sat_objs[1]), 
        min(pat_sat_objs[2], total_appts_objs[2])
    ]

    delta_eps = [None, 1, 0.1]

    EPS1Con = m.addConstr(objective_1 >= 0)
    EPS2Con = m.addConstr(objective_2 >= 0)

    # Generate or load Pareto results
    model_names = {
        FEASIBILITY: "feasibility",
        COMPATIBLE_TIMES: "compatible_times",
        DOCTOR_AVAILABLE: "doctor_available"
    }
    path = "outputs/results"
    filename = (f"{path}/pareto_{model_names[model]}_seed{seed}_I{len(I)}_J{len(J)}_K{len(K)}_T{len(T)}.pkl")

    if os.path.exists(filename):
        # Load previously saved results
        with open(filename, "rb") as f:
            output = pickle.load(f)
        print(f"[INFO] Successfully loaded existing results from {filename}")

        pareto_slack = output.get("pareto_slack")
        dom_slack = output.get("dom_slack")

        if dense:
            pareto_dense = output.get("pareto_dense")
            dom_dense = output.get("dom_dense")

    else:
        # Compute both sets if file does not exist
        print("[INFO] Generating new Pareto frontier...")

        start_time = time.time()
        print("* Slack Pareto frontier")
        pareto_slack, dom_slack, pareto_ind_slack, dom_ind_slack = compute_pareto_set(m, Y, objectives, data, model, initial_lower_bound, initial_upper_bound, EPS1Con, EPS2Con, delta_eps, use_slack=True, verbose=True)
        slack_time = time.time() - start_time

        if dense:
            print("* Dense Pareto frontier")
            start_time = time.time()
            pareto_dense, dom_dense, pareto_ind_dense, dom_ind_dense = compute_pareto_set(m, Y, objectives, data, model, initial_lower_bound, initial_upper_bound, EPS1Con, EPS2Con, delta_eps, use_slack=False, verbose=True)
            dense_time = time.time() - start_time

        output = {
            "pareto_slack": pareto_slack,
            "dom_slack": dom_slack,
            "pareto_ind_slack": pareto_ind_slack,
            "dom_ind_slack": dom_ind_slack,
            "slack_time": slack_time,
        }

        if dense:
            dense_output = {
                "pareto_dense": pareto_dense,
                "dom_dense": dom_dense,
                "pareto_ind_dense": pareto_ind_dense,
                "dom_ind_dense": dom_ind_dense,
                "dense_time": dense_time,
            }
            output.update(dense_output)

        with open(filename, "wb") as f:
            pickle.dump(output, f)
        print(f"[INFO] Saved pareto frontier to {filename}")