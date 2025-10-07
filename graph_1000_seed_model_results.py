import pickle
import matplotlib.pyplot as plt
import numpy as np

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


