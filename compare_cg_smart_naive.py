import os
import pickle
import subprocess
import matplotlib.pyplot as plt

i_s = [5, 10, 15, 20, 25]
js = [5]
ts = [20]
ks = [2]

def get_data(type, i, j, t, k):
    """
    Loads or generates schedule data.

    Parameters
    ----------
    type : str
        Type label used in the output filename (e.g., 'full', 'reduced').
    i, j, t, k : int
        Problem parameters.

    Returns
    -------
    data : any
        The unpickled data from cg_{type}_output_I{i}_J{j}_T{t}_K{k}.pkl
    """

    output_file = f"cg_{type}_output_I{i}_J{j}_T{t}_K{k}.pkl"
    data_file = f"data_seed10_I{i}_J{j}_K{k}_T{t}.pkl"

    # Try to open existing output
    if not os.path.exists(output_file):
        print(f"[INFO] Output file {output_file} not found.")

        # Step 1: Check if data for B exists, otherwise run program A
        if not os.path.exists(data_file):
            print(f"[INFO] Data file {data_file} not found. Running data_gen.py...")
            subprocess.run(["python", "data_gen.py", str(i), str(j), str(k), str(t)], check=True)
        else:
            print(f"[INFO] Found data file {data_file}.")

        # Step 2: Run program B to generate the missing output
        print(f"[INFO] Running cg_schedules_timed.py for arguments i={i}, j={j}, k={k}, t={t}...")
        subprocess.run(
            ["python", "cg_schedules_timed.py", f"data_seed10_I{i}_J{j}_K{k}_T{t}.pkl"],
            check=True
        )

    # Step 3: Load the output data
    print(f"[INFO] Loading {output_file}...")
    with open(output_file, "rb") as f:
        data = pickle.load(f)

    print(f"[INFO] Successfully loaded data for type={type}, I={i}, J={j}, T={t}, K={k}.")
    return data







def save_data(type: str, i_s=i_s, js=js,ts=ts,ks=ks):
    data = {}
    for i in i_s:
        for j in js:
            for t in ts:
                for k in ks:
                    data[i,j,t,k] = get_data(type, i, j, t, k)
    return data
    

# print(data["I"])
smart_data = save_data("smart")
smart_gen_times_for_each_patient_size = [smart_data[i,js[0],ts[0],ks[0]]["time_taken_for_doctor"] for i in i_s]
for ii, num_patients_data in enumerate(smart_gen_times_for_each_patient_size):
    print(i_s[ii],num_patients_data)
avg_smart_gen_time_per_patient_size = [sum(separate_doctor_times) for separate_doctor_times in smart_gen_times_for_each_patient_size]
#  Average generation time for each doctor with different numbers of patients
plt.scatter(i_s,avg_smart_gen_time_per_patient_size, s=100, marker='x')
plt.title("Average column generation time for different problem sizes")
# plt.grid(True)
_, right = plt.xlim()
plt.xticks([0] + i_s)
plt.xlim((0, right))
plt.xlabel("Number of patients")
plt.ylabel("Average time taken per doctor (s)")
# plt.margins(0,0)
plt.savefig("smart_cg_times")
plt.show()
# globals().update(data)


# doctor_gen_times = [[smart_gen_times_for_each_patient_size[s][doctor] for s in range(len(i_s))] for doctor in range(js[0])]
# scaled_gen_doctor_times = [[smart_gen_times_for_each_patient_size[s][doctor]/smart_gen_times_for_each_patient_size[-1][doctor] for s in range(len(i_s))] for doctor in range(js[0])]

# for doctor in range(js[0]):
#     plt.scatter(i_s,scaled_gen_doctor_times[doctor])
# plt.title("Smart times")
# plt.grid(True)
# _, right = plt.xlim()
# plt.xticks([0] + i_s)
# plt.xlim((0, right))
# plt.xlabel("Number of patients |I|")
# plt.ylabel("time taken for each doctor, scaled")
# plt.margins(0,0)
# plt.show()
# # globals().update(data)