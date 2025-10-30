import pickle

from utils.data_instance import DataInstance

START_TIME = 0
PATIENT_LIST = 1
NEXT_AVAILABLE_TIME = 2
PATIENT_TIME_LIST = 3

def patient_can_be_added_to_fragment(this_compat_times, patient, doctor, fragment):
    if patient in fragment[PATIENT_LIST]:
        return False

    return fragment[NEXT_AVAILABLE_TIME] in this_compat_times

def gen_fragments_for_doctor(d: DataInstance, doctor: int, max_frag_length: int):
    T = d.T
    fragments_length_n = dict()
    fragments_length_n[0] =  []
    patients = d.patients_doctor_can_treat[doctor]

    fragments_length_n[1] = [(start_time, (patient,), start_time + d.treat[doctor][d.patient_diseases[patient]], ((patient, start_time),)) for patient in patients for start_time in d.compatible_times[patient, doctor]]
    print(1, len(fragments_length_n[1]))
    max_length_fragments_grouped_by_next_available_time = {t: [] for t in T}
    fragments_by_start_time = {t: [] for t in T}
    for n in range(2, max_frag_length + 1):
        fragments_length_n[n] = []
        for fragment in fragments_length_n[n-1]:
            for patient in patients:
                if patient_can_be_added_to_fragment(d.compatible_times[patient, doctor], patient, doctor, fragment):
                    new_frag = (fragment[START_TIME], 
                                                  fragment[PATIENT_LIST] + (patient,), 
                                                  fragment[NEXT_AVAILABLE_TIME] + d.treat[doctor][d.patient_diseases[patient]],
                                                  fragment[PATIENT_TIME_LIST] + ((patient, fragment[NEXT_AVAILABLE_TIME]),)                                                  
                                                  )
                    fragments_by_start_time[new_frag[START_TIME]].append(new_frag)
                    if n == max_frag_length and new_frag[NEXT_AVAILABLE_TIME] < len(T):
                        max_length_fragments_grouped_by_next_available_time[new_frag[NEXT_AVAILABLE_TIME]].append(new_frag)

                    fragments_length_n[n].append(new_frag)
        print(n, len(fragments_length_n[n]))
    
    allFragments = []
    for n in range(1, max_frag_length + 1):
        allFragments.extend(fragments_length_n[n])

    return allFragments, max_length_fragments_grouped_by_next_available_time, fragments_by_start_time

def normal_generate_fragments(d: DataInstance, max_frag_length: int, save_output=True):
    F = dict()
    max_length_fragments_by_next_time = dict()
    fragments_by_start_time = dict()
    for j in d.J:
        print(f"doctor: {j}, diseases: {d.diseases_doctor_qualified_for[j]}, treat times: {[d.treat[j][k] for k in d.diseases_doctor_qualified_for[j]]}, length available: {d.doctor_available[j][1]}, ")
        F[j], max_length_fragments_by_next_time[j], fragments_by_start_time[j] = gen_fragments_for_doctor(d, j, max_frag_length)

    data = {
        "F": F,
        "max_length_fragments_by_next_time": max_length_fragments_by_next_time,
        "fragments_by_start_time": fragments_by_start_time,
        "I": d.I,
        "J": d.J,
        "T": d.T,
        "treat": d.treat,
        "patient_diseases": d.patient_diseases,
        "doctor_times": d.doctor_times,
    }

    if save_output:
        with open(f"data/cg_fragments_maxfraglength{max_frag_length}_seed{d.seed}_I{len(d.I)}_J{len(d.J)}_T{len(d.T)}_K{len(d.K)}.pkl", "wb") as f:
            pickle.dump(data, f)
    
    return data