import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 16
})

def extract_runtimes(model_results):
    runtimes = [res["model_results"]["stats"]["runtime"] for res in model_results.values()]
    runtimes = np.array(runtimes)
    runtimes.sort()
    y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100  # percentage
    return runtimes, y

def plot_performance_profiles(all_models, objectives, model_styles):
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
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Solved instances (%)")
        ax.set_title(obj_label)

    axes[-2].legend(title="Model:", loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig("outputs/graphs/graph_1000_seed_model_results.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_pareto_runtime_performance(model_runtimes, model_styles, model_names, label="Pareto", outfile="graph_pareto_runtime.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, runtimes in model_runtimes.items():
        if runtimes:
            runtimes = np.sort(np.array(runtimes))
            y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100
            ax.plot(runtimes, y, label=model_names[model_name], **model_styles.get(model_name, {}))
        else:
            print(f"⚠ No data found for: {model_name}")

    ax.set_xscale("log")
    xticks = [0.1, 1, 10, 100]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t}s" for t in xticks])

    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Solved instances (%)")
    ax.set_title(f"{label} performance profiles")
    ax.legend(title = "Model:", loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.show()

def plot_model_files(model_files, files):
    all_models = {}
    for model_name, file_path in model_files.items():
        with open(file_path, "rb") as f:
            all_models[model_name] = pickle.load(f)

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

    elif files == 2:
        fig, ax = plt.subplots(figsize=(8, 6))

        name_map = {
            "Singular:": "Serial",
            "Multiproccessing: ": "Multi-processing",
        }

        for model_name, model_results in all_models.items():
            display_name = name_map[model_name]

            runtimes = [res["runtime_seconds"] for res in model_results.values()]
            runtimes = np.array(runtimes)
            runtimes.sort()
            y = np.arange(1, len(runtimes) + 1) / len(runtimes) * 100
            ax.plot(runtimes, y, label=display_name)

        ax.set_xscale("log")
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Solved instances (%)")
        ax.set_title("Column generation performance profiles")
        ax.legend(title="Procedure:", loc="lower right", frameon=False)
        plt.tight_layout()
        plt.savefig("outputs/graphs/graph_runtime_singular_vs_multiprocessing.png", bbox_inches="tight", dpi=300)
        plt.show()

    elif (files == 3):
        seeds = range(1, 101)
        patients, doctors, diseases, time_periods = 50, 5, 4, 20

        model_styles = {
            "feasibility": {"color": "blue", "linestyle": "-"},
            "compatible_times": {"color": "green", "linestyle": "-"},
            "doctor_available": {"color": "red", "linestyle": "-"},
        }

        model_names = {
            "feasibility": "Feasibility",
            "compatible_times": "Compatible times",
            "doctor_available": "Doctor available",
        }

        # Collect runtimes for all models
        pareto_runtimes = {model: [] for model in model_styles.keys()}

        for seed in range(1,101):
            for model in model_styles.keys():
                filename = f"outputs/results/pareto_{model}_seed{seed}_I{50}_J{5}_K{4}_T{20}.pkl"
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        data = pickle.load(f)
                    pareto_runtimes[model].append(data.get("total_runtime", 0))
                else:
                    print(f"[WARN] Missing: {filename}")

        plot_pareto_runtime_performance(pareto_runtimes, model_styles, model_names, outfile="outputs/graphs/graph_pareto_runtime_all_models.png")