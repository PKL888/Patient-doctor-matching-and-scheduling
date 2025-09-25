import random
import math

def gen_best(K):
    return [random.choice([1,2,3]) for k in K]
    # return [3 for k in K]

def gen_treat(J, K, best):
    treatments = []
    for j in J:
        row = []
        for k in K:
            # keep drawing until we get a valid value in (0, 1]
            denom = 0
            while denom <= 0 or denom > 1:
                denom = random.normalvariate(0.75, 0.5)
            row.append(math.ceil(best[k] / denom))
        treatments.append(row)
    return treatments

# def gen_treat(J, K, best):
#     treat = []
#     for j in J:
#         while True:
#             # generate random treat times
#             times = [math.ceil(best[k] / min(max(0.000001, random.normalvariate(0.75, 0.5)), 1)) for k in K]
#             # check if doctor can treat at least one disease
#             if any(t <= len(T) for t in times):
#                 treat.append(times)
#                 break
#     return treat

# Checks if a doctor can treat a disease based on the number of time periods available
def gen_qualified(T, treat):
    return [[treat_time <= len(T) for treat_time in j] for j in treat]

M1 = 1e6

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
        min_time = max((treat[j][k] for k in K if qualified[j][k]), default=0)
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
    # wants to
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

import json

if __name__ == "__main__":
    NUM_SEEDS = 1000

    problem_size = {
        "patients": 100,
        "doctors":  10,
        "diseases": 4,
        "time periods": 20
    }

    ALL_DATA = {}

    for SEED in range(NUM_SEEDS):
        random.seed(SEED)

        I = range(problem_size["patients"])
        J = range(problem_size["doctors"])
        K = range(problem_size["diseases"])
        T = range(problem_size["time periods"])

        best = gen_best(K)
        treat = gen_treat(J, K, best)
        qualified = gen_qualified(T, treat)

        doctor_rank = gen_doctor_rank(qualified)
        doctor_available = gen_doctor_available(J, K, T, qualified, treat)
        patient_diseases = gen_patient_diseases(I, K)
        allocate_rank = gen_allocate_rank(I, J, patient_diseases, qualified)
        patient_available = gen_patient_available(I, J, T, patient_diseases, qualified, treat)
        patient_time_prefs = gen_patient_time_prefs(I, T, patient_available)

        # Bundle into a dictionary
        data = {
            "problem_size": problem_size,
            "best": best,
            "treat": treat,
            "qualified": qualified,
            "doctor_rank": doctor_rank,
            "doctor_available": doctor_available,
            "patient_diseases": patient_diseases,
            "allocate_rank": allocate_rank,
            "patient_available": patient_available,
            "patient_time_prefs": patient_time_prefs
        }

        # Save under this seed
        ALL_DATA[f"seed_{SEED}"] = data

    # Save to JSON
    with open(f"all_data_{NUM_SEEDS}_seeds_I{problem_size['patients']}_J{problem_size['doctors']}_K{problem_size['diseases']}_T{problem_size['time periods']}.json", "w") as f:
        json.dump(ALL_DATA, f, indent=4)
        