import random
import math

def gen_best(K):
    return [random.choice([1,2,3]) for k in K]

def gen_treat(J, K, best):
    return [[math.ceil(best[k] / min(max(0.000001,random.normalvariate(0.75,0.5)),1)) for k in K] for j in J]

# Checks if a doctor can treat a disease based on the number of time periods available
def gen_qualified(T, treat):
    return [[treat_time <= len(T) for treat_time in j] for j in treat]

M1 = 100

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
        start_time = random.choice(range(1, num_time_periods - length_available + 2))
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
        start_time = random.choice(range(1, num_time_periods - length_available + 2))
        ans.append((start_time, length_available))
    return ans

if __name__ == "__main__":

    I = range(10)
    J = range(4)
    K = range(2)
    T = range(8)

    random.seed(10)

    best = gen_best(K)
    treat = gen_treat(J, K, best)
    qualified = gen_qualified(T, treat)

    print("Best treatment times:", best)
    print("Doctor service times:",treat)
    print("Enough time to treat:", qualified)

    doctor_rank = gen_doctor_rank(qualified)
    doctor_available = gen_doctor_available(J, K, T, qualified, treat)
    
    print("-" * 21)
    print("Disease rank by doct:", doctor_rank)
    print("Doctor start, length:", doctor_available)

    patient_diseases = gen_patient_diseases(I, K)
    allocate_rank = gen_allocate_rank(I, J, patient_diseases, qualified)
    patient_available = gen_patient_available(I, J, T, patient_diseases, qualified, treat)

    print("-" * 21)
    print("Diseases by patients:", patient_diseases)
    print("Doctor rank by patie:", allocate_rank)
    print("Patient start, lengt:", patient_available)