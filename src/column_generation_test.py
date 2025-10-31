import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time

from huge.cg_fragments_num_appointments_generation import normal_generate_fragments
from utils.data_gen import get_data
from utils.data_instance import DataInstance

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 16
})

i_s = np.arange(10, 210, 10)
js = [10]
ts = [20]
ks = [2]
ns = [2, 3, 4, 5]

path = f"outputs/results"
filename = f"{path}/times_for_is_{len(i_s)=}_{js=}_{ts}_{ks=}_{len(ns)=}.pkl"

# --- Step 1: Compute or load results incrementally ---
if os.path.exists(filename):
    with open(filename, "rb") as f:
        times_for_is = pickle.load(f)
else:
    times_for_is = {n: [] for n in ns}

    for i in i_s:
        print(f"\nDoing i = {i}")
        # Get data
        problem_size = {
            "patients": i,
            "doctors": js[0],
            "diseases": ks[0],
            "time periods": ts[0]
        }
        all_data = get_data(problem_size, [11])
        data = all_data[f"seed_11"]

        d = DataInstance(data)

        for n in ns:
            if n >= 4 and i >= 60:
                break
            print(f"i: {i}, n: {n}")
            start_time = time.perf_counter()
            normal_generate_fragments(d, n, save_output=True)
            time_genning = time.perf_counter() - start_time
            times_for_is[n].append(time_genning)

        # Save after each i
        with open(filename, "wb") as f:
            pickle.dump(times_for_is, f)
        print(f"Saved progress for i={i}")

# --- Step 2: Plot ---
def plot_fragment_generation_times(times_for_is: dict[int, list[float]], i_s: list[int]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left: i < 60, all ns
    ax = axes[0]
    filtered_idx = [idx for idx, i in enumerate(i_s) if i < 60]
    filtered_i_s = [i_s[idx] for idx in filtered_idx]

    for n in ns:
        filtered_times = [times_for_is[n][idx] for idx in filtered_idx]
        ax.plot(filtered_i_s, filtered_times, marker="o", label=f"n={n}", linewidth=2)
    ax.set_yscale("log")
    ax.set_title("All fragment lengths")
    ax.set_xlabel("Number of patients")
    ax.set_ylabel("Generation time (s)")
    ax.legend(title="Fragment lengths:")

    # Right: i >= 60, only n=2,3
    ax = axes[1]
    filtered_idx = [idx for idx, i in enumerate(i_s) if i >= 60]
    filtered_i_s = [i_s[idx] for idx in filtered_idx]

    for n in [2, 3]:
        filtered_times = [times_for_is[n][idx] for idx in filtered_idx]
        ax.plot(filtered_i_s, filtered_times, marker="o", label=f"n={n}", linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Short fragment lengths")
    ax.set_xlabel("Number of patients")
    ax.legend(title="Fragment lengths:")

    plt.tight_layout()
    plt.savefig(f"outputs/graphs/fragment_gen_times_log_split.png", bbox_inches="tight", dpi=300)
    plt.show()

# --- Load and plot ---
with open(filename, "rb") as f:
    times = pickle.load(f)

plot_fragment_generation_times(times, i_s)