import pickle
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# code for comparing 3 basic model performance
# ==============================

# --- Helper Function ---
def extract_runtimes(model_results):
    runtimes = [res["model_results"]["stats"]["runtime"] for res in model_results.values()]
    runtimes = np.array(runtimes)
    runtimes.sort()
    y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100  # percentage
    return runtimes, y

# --- Load all 3 model result files ---
model_files = {
    "Feasibility": "F_all_1000_seeds_model_results.pkl",
    "Doctor_available": "DA_all_1000_seeds_model_results.pkl",
    "Compatible_times": "CT_all_1000_seeds_model_results.pkl",
}

all_models = {}
for model_name, file_path in model_files.items():
    with open(file_path, "rb") as f:
        all_models[model_name] = pickle.load(f)

# --- Define objectives to compare ---
objectives = {
    "max_matches": "Max Matches",
    "patient_satisfaction": "Patient Satisfaction",
    "doctor_satisfaction": "Doctor Satisfaction"
}

# --- Colors/linestyles for each model ---
styles = {
    "Feasibility": {"color": "blue", "linestyle": "-"},
    "Doctor_available": {"color": "orange", "linestyle": "--"},
    "Compatible_times": {"color": "green", "linestyle": ":"},
}

# plot all 3 in one window
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (obj_key, obj_label) in enumerate(objectives.items()):
    ax = axes[i]
    for model_name, model_results in all_models.items():
        x, y = extract_runtimes(model_results[obj_key])
        ax.plot(x, y, label=model_name, **styles[model_name])
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Solved Instances (%)")
    ax.set_title(obj_label)
    ax.grid(True, which="both", ls="--")
ax.legend(title="Model", loc="lower right")
plt.tight_layout()
plt.show()

# plot all 3 graphs in seperate windows
# for obj_key, obj_label in objectives.items():
#     plt.figure(figsize=(10, 6))
#     for model_name, model_results in all_models.items():
#         x, y = extract_runtimes(model_results[obj_key])
#         plt.plot(x, y, label=model_name, **styles[model_name])
    
#     plt.xscale("log")
#     plt.xlabel("Runtime (s)")
#     plt.ylabel("Solved Instances (%)")
#     plt.title(f"Runtime Distribution Comparison – {obj_label}")
#     plt.legend(title="Model")
#     plt.grid(True, which="both", ls="--")
#     plt.tight_layout()
#     plt.show(block=False)  # ← show without blocking

# input("Press Enter to close all plots...")

# ===================================================
# Code for comparing column generation run times
# ===================================================


# --- Helper Function ---
# def extract_runtimes(model_results):
#     """Extract sorted runtimes and cumulative percentage of solved seeds."""
#     runtimes = [res["runtime_seconds"] for res in model_results.values()]
#     runtimes = np.array(runtimes)
#     runtimes.sort()
#     y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100  # percentage
#     return runtimes, y

# # --- Load two CG files ---
# model_files = {
#     "Serial": "cg_schedules_timed_all_100_seeds_model_results.pkl",
#     "Multi-processing": "cg_schedules_timed_multiproccessing_all_100_seed_model_results.pkl"
# }

# all_models = {}
# for name, path in model_files.items():
#     with open(path, "rb") as f:
#         all_models[name] = pickle.load(f)

# # --- Colors / styles ---
# styles = {
#     "Serial": {"color": "blue", "linestyle": "-"},
#     "Multi-processing": {"color": "orange", "linestyle": "--"},
# }

# plt.rcParams.update({
#     "mathtext.fontset": "cm",   # Computer Modern for math
#     "font.family": "serif",     # Serif font for text
#     "font.size": 12            # Set default font size to 12
# })

# # --- Objectives to compare (can add more if available in your results) ---
# objectives = {
#     "runtime": "Runtime Distribution",
# }

# # --- Plot 3 objectives in one figure ---
# def plot_performance_profiles():
#     """Plots one subplot per objective, comparing models."""
#     plt.plot(figsize=(8, 5))

#     for model_name, model_results in all_models.items():
#         x, y = extract_runtimes(model_results)
#         plt.plot(x, y, label=model_name, **styles[model_name])

#     plt.xticks([0.1, 1, 10])

#     plt.xscale("log")
#     plt.xlabel("Runtime (s)")
#     plt.ylabel("Solved instances (%)")
#     plt.title("Column generation performance profile")
#     #plt.grid(True, which="both", ls="--")
#     plt.legend(title="Procedures: ", loc="lower right", frameon=False)
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig("Multi-processing.png")


# plot_performance_profiles()