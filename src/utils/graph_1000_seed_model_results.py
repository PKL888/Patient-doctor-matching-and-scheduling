import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern for math
    "font.family": "serif",     # Serif font for text
    "font.size": 16             # Set default font size to 12
})

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
    "Compatible times": "CT_all_1000_seeds_model_results.pkl",
    "Doctor available": "DA_all_1000_seeds_model_results.pkl",
}

all_models = {}
for model_name, file_path in model_files.items():
    with open(file_path, "rb") as f:
        all_models[model_name] = pickle.load(f)

# --- Define objectives to compare ---
objectives = {
    "patient_satisfaction": "Patient satisfaction",
    "max_matches": "Total appointments",
    "doctor_satisfaction": "Doctor satisfaction"
}

# --- Colors/linestyles for each model ---
model_styles = {
    "Feasibility": {"color": "blue", "linestyle": "-"},
    "Compatible times": {"color": "green", "linestyle": "-"},
    "Doctor available": {"color": "red", "linestyle": "-"},
}

obj_styles = {
    "patient_satisfaction": {"color": "blue", "linestyle": "-"},
    "max_matches": {"color": "green", "linestyle": "-"},
    "doctor_satisfaction": {"color": "red", "linestyle": "-"},
}
# plot all 3 in one window
def plot_performance_profiles():
    """Plots one subplot per objective, comparing models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (obj_key, obj_label) in enumerate(objectives.items()):
        ax = axes[i]
        for model_name, model_results in all_models.items():
            x, y = extract_runtimes(model_results[obj_key])
            ax.plot(x, y, label=model_name, **model_styles[model_name])

        # Log scale for x-axis
        ax.set_xscale("log")

        # Log-spaced ticks and labels
        xticks = [0.1, 1, 10]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t}s" for t in xticks])

        # Labels and title
        ax.set_xlabel("Runtime")
        ax.set_ylabel("Solved instances (%)")
        ax.set_title(obj_label)

    axes[-2].legend(title="Model:", loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig("graph_1000_seed_model_results.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_model_performance_comparison():
    """Plots one subplot per model, showing results for all objectives."""
    fig, axes = plt.subplots(1, len(all_models), figsize=(16, 5))

    for i, (model_name, model_results) in enumerate(all_models.items()):
        ax = axes[i]
        for obj_key, obj_label in objectives.items():
            x, y = extract_runtimes(model_results[obj_key])
            ax.plot(x, y, label=obj_label, **obj_styles[obj_key])

        # Log scale for x-axis
        ax.set_xscale("log")

        # Log-spaced ticks and labels
        xticks = [0.1, 1, 10]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t}s" for t in xticks])

        # Labels and title
        ax.set_xlabel("Runtime")
        ax.set_ylabel("Solved instances (%)")
        ax.set_title(model_name)

    axes[-1].legend(title="Objective", loc="lower right")
    plt.tight_layout()
    plt.savefig("graph_1000_seed_model_results_by_model.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

plot_performance_profiles()
# for obj_key, obj_label in objectives.items():
#     plt.figure(figsize=(10, 6))
#     for model_name, model_results in all_models.items():
#         x, y = extract_runtimes(model_results[obj_key])
#         plt.plot(x, y, label=model_name, **model_styles[model_name])
    
#     plt.xscale("log")
#     plt.xlabel("Runtime (s)")
#     plt.ylabel("Solved Instances (%)")
#     plt.title(f"Runtime Distribution Comparison – {obj_label}")
#     plt.legend(title="Model")
#     plt.grid(True, which="both", ls="--")
#     plt.tight_layout()
#     plt.show(block=False)  # ← show without blocking

# input("Press Enter to close all plots...")


