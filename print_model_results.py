import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

# ===============================================================
# STEP 1 — Summarize a model’s results
# ===============================================================
def summarize_model(all_results, model_name):
    summary = {
        "model_name": model_name,
        "before_presolve": {},
        "objectives": {}
    }

    # ---- BEFORE presolve ----
    if "before_presolve_info" in all_results:
        info = all_results["before_presolve_info"]
        summary["before_presolve"] = {
            "Variables": info["num_variables"],
            "Constraints": info["num_constraints"],
            "Nonzeros": info["num_nonzeros"],
            "Setup Time (s)": info["setup_time_seconds"]
        }
    else:
        before_info = {"num_variables": [], "num_constraints": [], "num_nonzeros": [], "setup_time_seconds": []}
        for obj_data in all_results.values():
            if not isinstance(obj_data, dict):
                continue
            for seed_data in obj_data.values():
                if "before_presolve_info" in seed_data:
                    b = seed_data["before_presolve_info"]
                    for k in before_info:
                        before_info[k].append(b[k])
        if before_info["num_variables"]:
            summary["before_presolve"] = {
                "Variables": str(int(np.mean(before_info["num_variables"]))),
                "Constraints": str(int(np.mean(before_info["num_constraints"]))),
                "Nonzeros": str(int(np.mean(before_info["num_nonzeros"]))),
                "Setup Time (s)": round(np.mean(before_info["setup_time_seconds"]), 3)
            }

    # ---- OBJECTIVE sections ----
    for obj_name, seeds_data in all_results.items():
        if obj_name == "before_presolve_info" or not isinstance(seeds_data, dict):
            continue

        runtimes, obj_values, pat_sat, doc_sat, num_alloc, appt_doc = ([] for _ in range(6))
        before_vars, before_cons, after_vars, after_cons = ([] for _ in range(4))

        for seed, data in seeds_data.items():
            stats = data["model_results"]["stats"]
            runtimes.append(stats["runtime"])
            obj_values.append(stats["objective_value"])
            pat_sat.append(stats["patient_satisfaction"])
            doc_sat.append(stats["doctor_satisfaction"])
            num_alloc.append(stats["num_patients_allocated"])
            appt_doc.append(stats["appointments_per_doctor"])

            if "before_presolve_info" in data:
                bp = data["before_presolve_info"]
                before_vars.append(bp["num_variables"])
                before_cons.append(bp["num_constraints"])

            if "after_presolve_info" in data["model_results"]:
                ap = data["model_results"]["after_presolve_info"]
                after_vars.append(ap.get("num_variables", np.nan))
                after_cons.append(ap.get("num_constraints", np.nan))

        cols_removed = np.array(before_vars) - np.array(after_vars)
        rows_removed = np.array(before_cons) - np.array(after_cons)

        summary["objectives"][obj_name] = {
            "Objective Value": f"{np.mean(obj_values):.2f} ± {np.std(obj_values):.2f}",
            "Patients Allocated": f"{np.mean(num_alloc):.2f} ± {np.std(num_alloc):.2f}",
            "Patient Satisfaction": f"{np.mean(pat_sat):.3f} ± {np.std(pat_sat):.3f}",
            "Doctor Satisfaction": f"{np.mean(doc_sat):.3f} ± {np.std(doc_sat):.3f}",
            "Appointments\nper Doctor": f"{np.mean(appt_doc):.2f} ± {np.std(appt_doc):.2f}",
            "Runtime (s)": f"{np.mean(runtimes):.2f} ± {np.std(runtimes):.2f}",
            "After Vars": f"{np.mean(after_vars):.1f} ± {np.std(after_vars):.1f}",
            "After Cons": f"{np.mean(after_cons):.1f} ± {np.std(after_cons):.1f}",
            "Cols Removed": f"{np.mean(cols_removed):.1f} ± {np.std(cols_removed):.1f}",
            "Rows Removed": f"{np.mean(rows_removed):.1f} ± {np.std(rows_removed):.1f}"
        }

    return summary


# ===============================================================
# STEP 2 — Load models
# ===============================================================
model_files = {
    "Feasibility model": "F_all_1000_seeds_model_results.pkl",
    "Compatible times model": "CT_all_1000_seeds_model_results.pkl",
    "Doctor available model": "DA_all_1000_seeds_model_results.pkl"
}

all_models = []
for model_name, file_path in model_files.items():
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    all_models.append(summarize_model(results, model_name))


# ===============================================================
# STEP 3 — Build comparison tables
# ===============================================================
before_data = pd.DataFrame({
    m["model_name"]: m["before_presolve"]
    for m in all_models
}).T

objective_tables = {}
for m in all_models:
    for obj_name, metrics in m["objectives"].items():
        df = pd.DataFrame(metrics, index=[m["model_name"]])
        if obj_name not in objective_tables:
            objective_tables[obj_name] = df
        else:
            objective_tables[obj_name] = pd.concat([objective_tables[obj_name], df])


# ===============================================================
# STEP 4 — Plot large figure (with overlap fixes)
# ===============================================================
fig_height = 2 + 3.3 * len(objective_tables)
fig, axes = plt.subplots(nrows=len(objective_tables) + 1, figsize=(15, fig_height))
if len(objective_tables) == 1:
    axes = [axes]

def create_table(ax, df, title, fontsize=9, scale_x=1.25, scale_y=1.4):
    ax.axis('off')
    # Wrap long headers
    wrapped_cols = ['\n'.join(wrap(c, 15)) for c in df.columns]
    tbl = ax.table(cellText=df.values,
                   rowLabels=df.index,
                   colLabels=wrapped_cols,
                   loc='center',
                   cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(scale_x, scale_y)
    ax.set_title(title, fontsize=13, pad=12)
    return tbl

# BEFORE presolve
create_table(axes[0], before_data, "BEFORE Presolve (Averaged over 1000 instances)", 
             fontsize=10, scale_x=1.3, scale_y=1.4)

# OBJECTIVES
for i, (obj_name, df) in enumerate(objective_tables.items(), start=1):
    create_table(axes[i], df, f"Objective: {obj_name}", 
                 fontsize=9, scale_x=1.25, scale_y=1.45)

plt.tight_layout(pad=3)
plt.subplots_adjust(hspace=0.6)
plt.savefig("model_comparison_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================================================
# STEP 5 — Save separate BEFORE presolve figure
# ===============================================================
fig2, ax2 = plt.subplots(figsize=(6.5, 1.25))
create_table(ax2, before_data, "", 
             fontsize=8, scale_x=0.75, scale_y=1.2)
plt.tight_layout(pad=1)
plt.savefig("before_presolve_summary.png", dpi=300, bbox_inches="tight")
plt.close()

#print("✅ Saved 'model_comparison_summary.png' and 'before_presolve_summary.png'")
