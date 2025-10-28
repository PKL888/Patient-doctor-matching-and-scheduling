import pickle
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from outputs.results import *
from src.huge import *

def summarize_results(all_results, model_name):
    """Aggregate presolve info and solver statistics across 100 seeds."""

    print("\n" + "="*100)
    print(f"MODEL SUMMARY: {model_name}")
    print("="*100)

    before_info_all = {
        "num_variables": [],
        "num_constraints": [],
        "num_nonzeros": [],
        "setup_time_seconds": []
    }

    found_before = False

    # --- BEFORE presolve info ---
    if "before_presolve_info" in all_results:
        presolve = all_results["before_presolve_info"]
        found_before = True
        print("\nBEFORE presolve info (shared across seeds):")
        print("-"*80)
        print(f"Number of variables:       {presolve['num_variables']}")
        print(f"Number of constraints:     {presolve['num_constraints']}")
        print(f"Number of nonzeros:        {presolve['num_nonzeros']}")
        print(f"Setup time (s):            {presolve['setup_time_seconds']:.4f}")

        before_vars = presolve["num_variables"]
        before_cons = presolve["num_constraints"]

    else:
        before_vars = []
        before_cons = []
        for obj_name, seeds_data in all_results.items():
            if not isinstance(seeds_data, dict):
                continue
            for seed, data in seeds_data.items():
                if "before_presolve_info" in data:
                    info = data["before_presolve_info"]
                    found_before = True
                    for k in before_info_all:
                        before_info_all[k].append(info[k])
                    before_vars.append(info["num_variables"])
                    before_cons.append(info["num_constraints"])

        if found_before and before_info_all["num_variables"]:
            print("\nAVERAGED BEFORE presolve info (across all seeds):")
            print("-"*80)
            for k, vals in before_info_all.items():
                arr = np.array(vals)
                print(f"{k:25s}: mean={np.mean(arr):.2f} ± {np.std(arr):.2f}")

    if not found_before:
        print("\n(No BEFORE presolve info found at any level.)")
        before_vars = before_cons = None

    # --- Process each objective ---
    for obj_name, seeds_data in all_results.items():
        if obj_name == "before_presolve_info" or not isinstance(seeds_data, dict):
            continue

        runtimes, obj_values, pat_sat, doc_sat, num_alloc, appt_doc = ([] for _ in range(6))
        after_info = {
            "columns_removed": [],
            "rows_removed": [],
            "num_variables": [],
            "num_constraints": [],
            "run_time_seconds": []
        }
        solver_nodes, solver_iters, solver_solutions, mip_gaps = ([] for _ in range(4))

        # --- Collect across all seeds ---
        for seed, data in seeds_data.items():
            stats = data["model_results"]["stats"]
            runtimes.append(stats["runtime"])
            obj_values.append(stats["objective_value"])
            pat_sat.append(stats["patient_satisfaction"])
            doc_sat.append(stats["doctor_satisfaction"])
            num_alloc.append(stats["num_patients_allocated"])
            appt_doc.append(stats["appointments_per_doctor"])

            if "mip_gap" in stats and stats["mip_gap"] is not None:
                mip_gaps.append(stats["mip_gap"])
            solver_nodes.append(stats["nodes"])
            solver_iters.append(stats["iterations"])
            solver_solutions.append(stats["solutions_found"])

            if "after_presolve_info" in data["model_results"]:
                ap = data["model_results"]["after_presolve_info"]
                for k in after_info:
                    after_info[k].append(ap[k])

        print("\n" + "="*80)
        print(f"Objective: {obj_name}")
        print("-"*80)

        # --- Objective-level averages ---
        print(f"Objective value (avg):     {np.mean(obj_values):.2f} ± {np.std(obj_values):.2f}")
        print(f"Patients allocated (avg):  {np.mean(num_alloc):.2f} ± {np.std(num_alloc):.2f}")
        print(f"Patient satisfaction:      {np.mean(pat_sat):.3f} ± {np.std(pat_sat):.3f}")
        print(f"Doctor satisfaction:       {np.mean(doc_sat):.3f} ± {np.std(doc_sat):.3f}")
        print(f"Appointments per doctor:   {np.mean(appt_doc):.2f} ± {np.std(appt_doc):.2f}")

        # --- Solver stats ---
        print("\nSolver stats (Gurobi, across 100 seeds):")
        print(f"Runtime (s):               min = {np.min(runtimes):.2f}, "
              f"mean={np.mean(runtimes):.2f}, max = {np.max(runtimes):.2f}, std={np.std(runtimes):.2f}")
        print(f"Nodes explored:            mean = {np.mean(solver_nodes):.1f} ± {np.std(solver_nodes):.1f}")
        print(f"Iterations:                mean = {np.mean(solver_iters):.1f} ± {np.std(solver_iters):.1f}")
        print(f"Solutions found:           mean = {np.mean(solver_solutions):.1f} ± {np.std(solver_solutions):.1f}")
        if mip_gaps:
            print(f"MIP gap:                   mean = {np.mean(mip_gaps):.4f} ± {np.std(mip_gaps):.4f}")

        # --- After presolve info ---
        if after_info["num_variables"]:
            print("\nAFTER presolve info (averaged across seeds):")
            for k, v in after_info.items():
                if (k == "columns_removed" or k == "rows_removed"):
                    continue
                else:
                    arr = np.array(v)
                    print(f"{k:25s}: mean = {np.mean(arr):.2f} ± {np.std(arr):.2f}")

            # --- Compute derived removals if before info is available ---
            if before_vars is not None and before_cons is not None:
                # Ensure before info matches the same number of seeds
                if isinstance(before_vars, list) and len(before_vars) > len(after_info["num_variables"]):
                    # Likely before_vars covers multiple objectives — use mean or unique values
                    before_vars_arr = np.full(len(after_info["num_variables"]), np.mean(before_vars))
                    before_cons_arr = np.full(len(after_info["num_constraints"]), np.mean(before_cons))
                else:
                    before_vars_arr = np.array(before_vars)
                    before_cons_arr = np.array(before_cons)

                vars_removed = before_vars_arr - np.array(after_info["num_variables"])
                cons_removed = before_cons_arr - np.array(after_info["num_constraints"])

                print(f"Columns removed (vars):    mean = {np.mean(vars_removed):.2f} ± {np.std(vars_removed):.2f}")
                print(f"Rows removed (constraints):mean = {np.mean(cons_removed):.2f} ± {np.std(cons_removed):.2f}")

        print("="*80)

    print("\nEND OF MODEL SUMMARY")
    print("="*100 + "\n")

import os


def summarize_pareto_slack_results(seeds, file_template="outputs/results/pareto_doctor_available_seed{seed}_I30_J3_K3_T10.pkl"):
    """
    Loop through saved seed files, extract Pareto slack info and slack times,
    and print summary statistics (mean, std, min, max).
    """
    num_solutions_slack = []
    num_dominated_slack = []
    slack_times = []

    for seed in seeds:
        filename = file_template.format(seed=seed)
        if not os.path.exists(filename):
            print(f"[WARNING] File {filename} not found, skipping.")
            continue

        with open(filename, "rb") as f:
            data = pickle.load(f)

        pareto_slack = data.get("pareto_slack") or []
        dom_slack = data.get("dom_slack") or []
        slack_time = data.get("slack_time")  # or data.get("total_runtime") if you saved runtime

        num_solutions_slack.append(len(pareto_slack))
        num_dominated_slack.append(len(dom_slack))
        if slack_time is not None:
            slack_times.append(slack_time)

    # Convert to numpy arrays
    num_solutions_slack = np.array(num_solutions_slack, dtype=float)
    num_dominated_slack = np.array(num_dominated_slack, dtype=float)
    slack_times = np.array(slack_times, dtype=float)

    def mean_std_min_max(arr):
        if len(arr) == 0:
            return np.nan, np.nan, np.nan, np.nan
        return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

    mean_s, std_s, min_s, max_s = mean_std_min_max(num_solutions_slack)
    mean_d, std_d, min_d, max_d = mean_std_min_max(num_dominated_slack)
    mean_t, std_t, min_t, max_t = mean_std_min_max(slack_times)

    print(f"Average number of Pareto-optimal solutions (slack): {mean_s:.2f} ± {std_s:.2f} (min={min_s}, max={max_s})")
    print(f"Average number of dominated solutions (slack):      {mean_d:.2f} ± {std_d:.2f} (min={min_d}, max={max_d})")
    print(f"Average slack computation time (s):                 {mean_t:.2f} ± {std_t:.2f} (min={min_t:.2f}, max={max_t:.2f})")
