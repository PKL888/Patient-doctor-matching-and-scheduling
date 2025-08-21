import random
import math



def gen_diseases(diseases):
    return [random.choice([1,2,3]) for i in diseases]

def gen_doctor_time_treat(doctors, diseases, disease_treat_dura):
    return [[math.ceil(disease_treat_dura[o] / min(max(0.000001,random.normalvariate(0.75,0.5)),1)) for o in diseases] for d in doctors]

def gen_doctor_can_treat(doctor_treat_times, time_periods):
    return [[doctor_treat_time <= len(time_periods)  for doctor_treat_time in doctor] for doctor in doctor_treat_times]

M1 = 100

def gen_doctor_disease_preferences(doctor_can_treat):
    ans = []
    for doctor in doctor_can_treat:
        num_diseases_to_rank = sum(doctor)
        disease_ranking = []
        ranks_to_give = [i for i in range(1, num_diseases_to_rank + 1)]
        random.shuffle(ranks_to_give)
        for disease_is_treated in doctor:
            if disease_is_treated:
                disease_ranking.append(ranks_to_give.pop(0))
            else:
                disease_ranking.append(M1)
        
        ans.append(disease_ranking)
    
    return ans

def gen_doctor_times(time_periods, diseases, doctors, doctor_treat_times, doctor_can_treat):
    num_time_periods = len(time_periods)
    ans = []
    for doctor in doctors:
        min_time = max(doctor_treat_times[doctor][k] for k in diseases if doctor_can_treat[doctor][k])
        length_available = random.choice(range(min_time, num_time_periods + 1))
        start_time = random.choice(range(1, num_time_periods - length_available + 2))
        ans.append((start_time, length_available))
    
    return ans




def gen_patient_diseases(patients, diseases):
    ans = []
    for patient in patients:
        disease = random.choice(diseases)
        ans.append(disease)
    return ans

def gen_patient_doctor_prefs(patients, doctors, patient_diseases, doctor_can_treat):
    ans = []
    for patient in patients:
        correct_doctors = [doctor for doctor in doctors if doctor_can_treat[doctor][patient_diseases[patient]]]
        num_doctors = len(correct_doctors)
        prefs = []
        ranks_to_give = [i for i in range(1, num_doctors + 1)]
        random.shuffle(ranks_to_give)
        for doctor in doctors:
            if doctor in correct_doctors:
                prefs.append(ranks_to_give.pop(0))
            else:
                prefs.append(M1)
        ans.append(prefs)
    return ans

def gen_patient_times(patients, doctors, time_periods, patient_diseases, doctor_can_treat, doctor_treat_times):
    # wants to
    num_time_periods = len(time_periods)

    ans = []
    for patient in patients:
        patient_disease = patient_diseases[patient]
        min_time = min(doctor_treat_times[doctor][patient_disease] for doctor in doctors if doctor_can_treat[doctor][patient_disease])

        length_available = random.choice(range(min_time, num_time_periods + 1))
        start_time = random.choice(range(1, num_time_periods - length_available + 2))
        ans.append((start_time, length_available))
    return ans

        

if __name__ == "__main__":

    O = range(2)
    T = range(8)
    D = range(4)
    P = range(10)


    random.seed(10)

    disease_treat_times = gen_diseases(O)
    doctor_treat_times = gen_doctor_time_treat(D, O, disease_treat_times)
    doctor_can_treat = gen_doctor_can_treat(doctor_treat_times, T)
    doctor_disease_prefs = gen_doctor_disease_preferences(doctor_can_treat)
    doctor_times = gen_doctor_times(T, O, D, doctor_treat_times, doctor_can_treat)

    print(doctor_treat_times)
    print(doctor_can_treat)
    print(doctor_disease_prefs)
    print(doctor_times)

    patient_diseases = gen_patient_diseases(P, O)
    patient_doctor_prefs = gen_patient_doctor_prefs(P, D, patient_diseases, doctor_can_treat)
    patient_times = gen_patient_times(P, D, T, patient_diseases, doctor_can_treat, doctor_treat_times)          

    print(patient_diseases)
    print(patient_doctor_prefs)
    print(patient_times)