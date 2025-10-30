import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern for math
    "font.family": "serif",     # Serif font for text
    "font.size": 16             # Set default font size
})

# --- Helper Function ---
def extract_runtimes(model_results):
    runtimes = [res["model_results"]["stats"]["runtime"] for res in model_results.values()]
    runtimes = np.array(runtimes)
    runtimes.sort()
    y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100  # percentage
    return runtimes, y


# --- Plot Functions ---
def plot_performance_profiles(all_models, objectives, model_styles):
    """Plots one subplot per objective, comparing models."""
    fig, axes = plt.subplots(1, len(objectives), figsize=(16, 5))

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
    plt.show()


def plot_model_performance_comparison(all_models, objectives, obj_styles):
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
    plt.show()

def plot_pareto_runtime_performance(results, label="Pareto Runtime", outfile="graph_pareto_runtime.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    runtimes = np.array(results)
    runtimes.sort()
    y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100

    ax.plot(runtimes, y, label=label)
    ax.set_xscale("log")

    xticks = [0.1, 1, 10, 100, 1000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t}s" for t in xticks])

    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_ylabel("Solved instances (%)")
    ax.set_title(f"Performance Profile: {label}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.show()



# --- Main Plot Function ---
def plot_model_files(model_files, files):
    """Loads models from pickle files and plots performance profiles."""
    all_models = {}
    for model_name, file_path in model_files.items():
        with open(file_path, "rb") as f:
            all_models[model_name] = pickle.load(f)

    # --- CASE 1: Compact models ---
    if files == 1:
        objectives = {
            "patient_satisfaction": "Patient satisfaction",
            "max_matches": "Total appointments",
            "doctor_satisfaction": "Doctor satisfaction"
        }

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

        plot_performance_profiles(all_models, objectives, model_styles)
        plot_model_performance_comparison(all_models, objectives, obj_styles)

    # --- CASE 2: Singular vs Multiprocessing runtime comparison ---
    elif files == 2:
        fig, ax = plt.subplots(figsize=(8, 6))

        for model_name, model_results in all_models.items():
            # collect total seed runtimes
            runtimes = [res["runtime_seconds"] for res in model_results.values()]
            runtimes = np.array(runtimes)
            runtimes.sort()
            y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100
            ax.plot(runtimes, y, label=model_name)

        ax.set_xscale("log")
        ax.set_xlabel("Runtime (seconds, log scale)")
        ax.set_ylabel("Solved seeds (%)")
        ax.set_title("Runtime Comparison: Singular vs Multiprocessing")
        ax.legend(title="Mode", loc="lower right")
        plt.tight_layout()
        plt.savefig("graph_runtime_singular_vs_multiprocessing.png", bbox_inches="tight", dpi=300)
        plt.show()

    elif (files == 3):
        seeds = range(1, 1001)
        patients, doctors, diseases, time_periods = 200, 20, 4, 20

        model_styles = {
            "feasibility": {"color": "blue", "linestyle": "-"},
            "compatible_times": {"color": "green", "linestyle": "-"},
            "doctor_available": {"color": "red", "linestyle": "-"},
        }

        pareto_runtimes = {model: [] for model in model_styles.keys()}

        for seed in seeds:
            for model in model_styles.keys():
                filename = f"outputs/results/pareto_{model}_seed{seed}_I{patients}_J{doctors}_K{diseases}_T{time_periods}.pkl"
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        data = pickle.load(f)
                    pareto_runtimes[model].append(data.get("total_runtime", 0))
                else:
                    print(f"[WARN] Missing: {filename}")

        for model_name, runtimes in pareto_runtimes.items():
            if runtimes:
                print(f"Plotting Pareto runtimes for: {model_name} ({len(runtimes)} instances)")
                plot_pareto_runtime_performance(runtimes, label=model_name, outfile=f"graph_pareto_runtime_{model_name}.png")
            else:
                print(f"⚠ No data found for: {model_name}")




