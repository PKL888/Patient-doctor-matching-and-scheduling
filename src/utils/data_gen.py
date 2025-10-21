import random
import math
import pickle

M1 = 1e6

def gen_best(K):
    return [random.choice([1,2,3]) for k in K]

def gen_treat(J, K, best):
    treatments = []
    for j in J:
        row = []
        for k in K:
            denom = 0
            while denom <= 0 or denom > 1:
                denom = random.normalvariate(0.75, 0.5)
            row.append(math.ceil(best[k] / denom))
        treatments.append(row)
    return treatments

def gen_qualified(T, treat):
    return [[treat_time <= len(T) for treat_time in j] for j in treat]

def gen_doctor_rank(qualified):
    ans = []
    for j in qualified:
        num_diseases_to_rank = sum(j)
        disease_ranking = []
        ranks_to_give = [i for i in range(1, num_diseases_to_rank + 1)]
        random.shuffle(ranks_to_give)
        for disease_is_treated in j:
            if disease_is_treated:
                disease_ranking.append(ranks_to_give.pop(0))
            else:
                disease_ranking.append(M1)
        ans.append(disease_ranking)
    return ans

def gen_doctor_available(J, K, T, qualified, treat):
    num_time_periods = len(T)
    ans = []
    for j in J:
        min_time = max(treat[j][k] for k in K if qualified[j][k])
        length_available = random.choice(range(min_time, num_time_periods + 1))
        start_time = random.choice(range(0, num_time_periods - length_available + 1))
        ans.append((start_time, length_available))
    return ans

def gen_patient_diseases(I, K):
    ans = []
    for i in I:
        disease = random.choice(K)
        ans.append(disease)
    return ans

def gen_allocate_rank(I, J, patient_diseases, qualified):
    ans = []
    for i in I:
        correct_doctors = [j for j in J if qualified[j][patient_diseases[i]]]
        num_doctors = len(correct_doctors)
        prefs = []
        ranks_to_give = [i for i in range(1, num_doctors + 1)]
        random.shuffle(ranks_to_give)
        for j in J:
            if j in correct_doctors:
                prefs.append(ranks_to_give.pop(0))
            else:
                prefs.append(M1)
        ans.append(prefs)
    return ans

def gen_patient_available(I, J, T, patient_diseases, qualified, treat):
    num_time_periods = len(T)
    ans = []
    for i in I:
        patient_disease = patient_diseases[i]
        min_time = min(treat[j][patient_disease] for j in J if qualified[j][patient_disease])
        length_available = random.choice(range(min_time, num_time_periods + 1))
        start_time = random.choice(range(0, num_time_periods - length_available + 1))
        ans.append((start_time, length_available))
    return ans

def gen_patient_time_prefs(I, T, patient_available):
    ans = []
    for i in I:
        prefs = []
        ranks_to_give = [i for i in range(1, patient_available[i][1] + 1)]
        random.shuffle(ranks_to_give)
        for t in T:
            if t in range(patient_available[i][0], patient_available[i][0] + patient_available[i][1]):
                prefs.append(ranks_to_give.pop(0))
            else:
                prefs.append(M1)
        ans.append(prefs)
    return ans

def generate_data(problem_size, num_seeds=1):
    """
    Generate doctor-patient problem datasets for a given problem size and number of seeds.
    Save to file.
    """
    all_data = {}

    for seed in range(num_seeds):
        random.seed(seed)

        I = list(range(problem_size["patients"]))
        J = list(range(problem_size["doctors"]))
        K = list(range(problem_size["diseases"]))
        T = list(range(problem_size["time periods"]))

        # Generate problem components
        best = gen_best(K)
        treat = gen_treat(J, K, best)
        qualified = gen_qualified(T, treat)
        doctor_rank = gen_doctor_rank(qualified)
        doctor_available = gen_doctor_available(J, K, T, qualified, treat)
        patient_diseases = gen_patient_diseases(I, K)
        allocate_rank = gen_allocate_rank(I, J, patient_diseases, qualified)
        patient_available = gen_patient_available(I, J, T, patient_diseases, qualified, treat)
        patient_time_prefs = gen_patient_time_prefs(I, T, patient_available)

        # Create binary time matrices
        def create_binary_times(availability_list, total_periods):
            binary_times = []
            for start, duration in availability_list:
                times = [1 if t in range(start, start + duration) else 0 for t in range(total_periods)]
                binary_times.append(times)
            return binary_times

        doctor_times = create_binary_times(doctor_available, len(T))
        patient_times = create_binary_times(patient_available, len(T))

        # Index sets
        START, DURATION = 0, 1
        I_k = {k: [i for i in I if patient_diseases[i] == k] for k in K}
        J_k = {k: [j for j in J if qualified[j][k]] for k in K}
        diseases_doctor_qualified_for = {j: [k for k in K if qualified[j][k]] for j in J}

        # Compute compatible times per (i,j) pair
        compatible_times = {}
        for k in K:
            for i in I_k[k]:
                for j in J_k[k]:
                    start_i = patient_available[i][START]
                    end_i = start_i + patient_available[i][DURATION]
                    start_j = doctor_available[j][START]
                    end_j = start_j + doctor_available[j][DURATION]
                    # Ensure treatment duration fits
                    end = max(0, min(end_i, end_j) - treat[j][k] + 1)
                    compatible_times[i, j] = list(range(max(start_i, start_j), end))

        # Compute patients each doctor can treat
        patients_doctor_can_treat = [
            [i for k in diseases_doctor_qualified_for[j] for i in I_k[k] if compatible_times[i, j]]
            for j in J
        ]

        # Compute patient and doctor scores
        numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
        patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
        patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]
        doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
        doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]         

        # Bundle data for this seed
        data = {
            "problem_size": problem_size,
            "seed": seed,
            "I": I,
            "J": J,
            "K": K,
            "T": T,
            "best": best,
            "treat": treat,
            "qualified": qualified,
            "doctor_rank": doctor_rank,
            "doctor_available": doctor_available,
            "patient_diseases": patient_diseases,
            "allocate_rank": allocate_rank,
            "patient_available": patient_available,
            "patient_time_prefs": patient_time_prefs,
            "doctor_times": doctor_times,
            "patient_times": patient_times,
            "START": START,
            "DURATION": DURATION,
            "I_k": I_k,
            "J_k": J_k,
            "diseases_doctor_qualified_for": diseases_doctor_qualified_for,
            "compatible_times": compatible_times,
            "patients_doctor_can_treat": patients_doctor_can_treat,
            "numberAvailableDoctors": numberAvailableDoctors,
            "patientDoctorScore": patientDoctorScore,
            "patientTimeScore": patientTimeScore,
            "doctor_num_diseases_can_treat": doctor_num_diseases_can_treat,
            "doctor_disease_rank_scores": doctor_disease_rank_scores
        }

        all_data[f"seed_{seed}"] = data

    path = "data"
    if num_seeds == 1:
        filename = f"{path}/data_seed{seed}_I{len(I)}_J{len(J)}_K{len(K)}_T{len(T)}.pkl"
    else:
        filename = f"{path}/all_data_{num_seeds}seeds_I{len(I)}_J{len(J)}_K{len(K)}_T{len(T)}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved all data to {filename}")

    
if __name__ == "__main__":
    num_seeds = 1

    problem_size = {
        "patients": 100,
        "doctors":  10,
        "diseases": 4,
        "time periods": 20
    }

    generate_data(problem_size, num_seeds)