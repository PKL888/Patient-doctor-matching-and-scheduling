from huge.cg_fragments_num_appointments_generation import normal_generate_fragments
from utils.data_gen import get_data
import time
from utils.data_instance import DataInstance
import matplotlib.pyplot as plt
import pickle
import os

# i_s = [10, 20, 30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
i_s = [10, 20, 30,40,50,60, 60,70,80]
js = [10]
ts = [20]
ks = [2]
# ns = [2,3]
ns = [2,3,4,5]

def test_fragments(i_s, j, t, k, ns, seed = 11):
    
    times_for_is = {n: [] for n in ns}
    for i in i_s:
        print(f"doing {i}")
        # get data
        problem_size = {
            "patients": i,
            "doctors":  j,
            "diseases": k,
            "time periods": t
        }
        all_data = get_data(problem_size, [seed])
        data = all_data[f"seed_{seed}"]

        d = DataInstance(data)
        
        for n in ns:
            print(f"i: {i}, n: {n}")
            start_time = time.perf_counter()
            normal_generate_fragments(d, n, save_output=True)
            time_genning = time.perf_counter() - start_time
            times_for_is[n].append(time_genning)
    return times_for_is




def plot_fragment_generation_times(times_for_is: dict[int, list[float]], i_s: list[int], log= False, bad_i_s = False):
    """
    Plots fragment generation time (seconds) vs number of patients (i_s)
    for each n (number of fragments).

    Parameters
    ----------
    times_for_is : dict[int, list[float]]
        Mapping from n → list of times for each i in i_s.
    i_s : list[int]
        List of patient counts corresponding to the times.
    """
    plt.figure(figsize=(8, 6))

    for n, times in times_for_is.items():
        if bad_i_s:
            plt.plot([i for (index, i) in enumerate(i_s) if index != 5], [t for (index, t) in enumerate(times) if index != 5], marker='o', label=f'n = {n}', linewidth=2)
        else:
            plt.plot(i_s, times, marker='o', label=f'n = {n}', linewidth=2)

    plt.title("Fragment Generation Time vs Number of patients")
    plt.xlabel("Number of patients")
    plt.ylabel("Generation Time (seconds)")
    plt.legend(title="Number of Fragments (n)")
    if log:
        plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"outputs/graphs/fragment_gen_times_plot_ns{ns[-1]}_is{i_s[-1]}_{log=}.png")
    plt.show()

filename = f"times_for_is_{i_s=},_{js=},{ts},{ks=},{ns=}"
if not os.path.exists(filename):
    times_for_is = test_fragments(i_s, js[0], ts[0], ks[0] ,ns)
    with open(filename, "wb") as f:
        data = pickle.dump(times_for_is, f)

with open(filename, "rb") as f:
    times = pickle.load(f)

plot_fragment_generation_times(times, i_s, log = False, bad_i_s=True)
plot_fragment_generation_times(times, i_s, log = True, bad_i_s=True)
