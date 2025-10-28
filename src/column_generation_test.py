from huge.cg_fragments_num_appointments_generation import normal_generate_fragments
from utils.data_gen import get_data
import time
from utils.data_instance import DataInstance
import matplotlib.pyplot as plt

i_s = [5, 10, 15, 20, 25,30,35,40,50,60]
js = [5]
ts = [20]
ks = [2]
ns = [2,3,4,5]

def test_fragments(i_s, j, t, k, ns, seed = 10):
    
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
            normal_generate_fragments(d, n, save_output=False)
            time_genning = time.perf_counter() - start_time
            times_for_is[n].append(time_genning)
    return times_for_is


times_for_is = test_fragments(i_s, js[0], ts[0], ks[0] ,ns)

def plot_fragment_generation_times(times_for_is: dict[int, list[float]], i_s: list[int]):
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
        plt.plot(i_s, times, marker='o', label=f'n = {n}', linewidth=2)

    plt.title("Fragment Generation Time vs Problem Size")
    plt.xlabel("Number of Patients (i)")
    plt.ylabel("Generation Time (seconds)")
    plt.legend(title="Number of Fragments (n)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

plot_fragment_generation_times(times_for_is, i_s)
