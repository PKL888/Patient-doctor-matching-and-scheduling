import gurobipy as gp
from data_gen import *
import random
import json
import time

file = "data_seed10_I5_J5_K2_T20.pkl"
print("Using", file)
with open(file, "rb") as f:
    data = pickle.load(f)
globals().update(data)


I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

"""
Only checks for blocks

appointment: (i,tstart, tend)
schedule

Returns True if the appointment can be added, False otherwise

"""
def appt_can_be_added(appointment: tuple, schedule: frozenset[tuple]):
    this_start, this_end = appointment[1], appointment[2]
    for appointment_in_schedule in schedule:
        if not (this_end <= appointment_in_schedule[1] or this_start >= appointment_in_schedule[2]):
            return False
    return True


S = dict()
print("starting to gen doctor schedules")
start = time.perf_counter()
time_gen_doctor = []
for j in J:
    start_doctor = time.perf_counter()
    print("Doctor:", j)
    print(f"doctor: {j}, diseases: {diseases_doctor_qualified_for[j]}, treat times: {[treat[j][k] for k in diseases_doctor_qualified_for[j]]}, length available: {doctor_available[j][1]}, ")
    schedules_with_size_n = dict()
    # Schedules_with_size_n[0] = {}
    schedules_with_size_n[1] = set(frozenset({(i,t,t+treat[j][k])}) for k in diseases_doctor_qualified_for[j] for i in I_k[k] for t in compatible_times[i,j])
    for upto in range(2, doctor_available[j][1] // min([treat[j][k] for k in diseases_doctor_qualified_for[j]]) + 1):
        print("upto:", upto)
        schedules_with_size_n[upto] = set()
        for schedule in schedules_with_size_n[upto - 1]:
            # print("|", end="")
            for appt in [(i,t, t+treat[j][k]) for k in diseases_doctor_qualified_for[j] for i in I_k[k] for t in compatible_times[i,j]]:
                # if can add this appointment
                if appt_can_be_added(appt, schedule):
                    # print("-", end = "")
                    schedules_with_size_n[upto].add(frozenset(schedule | {appt}))
        print("")
    time_gen_doctor.append(time.perf_counter()- start_doctor)
    total_time = time.perf_counter() - start
    print(f"Total time: {total_time:.2f} Doctor times:", [f"{t:.2f}" for t in time_gen_doctor])

    # print(type(schedules_with_size_n[1]))
    # print("size 1", len(schedules_with_size_n[1]), schedules_with_size_n[1])
    # print("size 2", len(schedules_with_size_n[2]), schedules_with_size_n[2])

    DoctorApptSets = set()
    for n in schedules_with_size_n:
        DoctorApptSets |= schedules_with_size_n[n]
    # DoctorApptSets = frozenset(frozenset(schedules_with_size_n[n]) for n in range(1,len(T)+1))
    # DoctorApptSets = frozenset(frozenset(schedules_with_size_n[n]) for n in range(1,len(T)+1))

    S[j] = DoctorApptSets
print("Genned doctor schedules")



# for j in J:
    # get answer from each thread / process


# print(f"Total wall-clock time in set generation:  {time_gen:.6f} s")
# print(f"Total time making and solving small MIPs: {time_mip:.6f} s")
# print(f"Total time making small MIPs:             {(time_mip - time_in_mip_solver):.6f} s")
# print(f"Total time solving small MIPs:            {time_in_mip_solver:.6f} s")
# print(f"Total MIP calls: {mip_calls}, feasible: {mip_feasible}")

# Save schedules to pickle, along with necessary variables to run the huge model and print the schedules
data = {
    "S": S,
    "I": I,
    "J": J,
    "T": T,
    "treat": treat,
    "patient_diseases": patient_diseases,
    "doctor_times": doctor_times,
    "time_taken_for_doctor": time_gen_doctor
}

with open(f"cg_naive_output_I{len(I)}_J{len(J)}_T{len(T)}_K{len(K)}.pkl", "wb") as f:
    pickle.dump(data, f)