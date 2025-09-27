import time

#################################################################
# printing and optimising
def left_pad_string(s, length):
    if len(s) >= length:
        return s
    
    return " " * (length - len(s)) + s 

def create_schedule(Ys, K, J, I_k, T, treat):
    schedule = []
    for j in J:
        # print("doctor:", j, "treatment length:", treat[j])
        # print("times available", doctor_times[j])
        # print("start appointment", [sum(Y[i,j,t].x for i in I) for t in T])
        # print("checking ", [sum((Y[i,j,tt].x) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) and not doctor_times[j][t] for t in T])
        doctor_schedule = [int(patient - 1) for patient in [sum((Ys[i,j,tt] * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]]
        # doctor_schedule_with_disease = [(patient, patient_diseases[patient]) for patient in doctor_schedule]
        # print("(patient, disease)", [(int(patient - 1), patient_diseases[int(patient - 1)]) for patient in [sum((Y[i,j,tt].x * (i + 1)) for k in K for i in I_k[k] for tt in T[max(0, t - treat[j][k] + 1):t+1]) for t in T]])
        # print("(patient, disease)", doctor_schedule_with_disease)
        schedule.append(doctor_schedule)
    return schedule

def print_stats(Ys, M1, I, J, K, T, I_k, allocate_rank, qualified, doctor_rank, patient_available, patient_time_prefs):
    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
    print("Stats: -----------------------------------")
    print("Number of patients allocated:", round(sum(Ys[i,j,t] for i in I for j in J for t in T)))
    print("Patient satisfaction with doctor and time:", round(sum(Ys[i,j,t] * ((numberAvailableDoctors[i] - allocate_rank[i][j] + 1)/numberAvailableDoctors[i] + ((patient_available[i][1]) + 1 - patient_time_prefs[i][t])/patient_available[i][1]) for i in I for j in J for t in T)))
    print("Doctor satisfaction with diseases:", round(sum((doctor_disease_rank_scores[j][k]) * Ys[i,j,t] for k in K for i in I_k[k] for j in J for t in T)))
    print("Appointments per doctor:", round(sum(Ys[i,j,t] for i in I for j in J for t in T))/len(J))

def print_schedule(schedule, I, J, T, doctor_times):
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))

    for j in J:
        formatted_doctor_schedule = []
        for t, patient in enumerate(schedule[j]):
            if patient is None or patient < 0:  # nobody scheduled
                if doctor_times[j][t]:  
                    s_val = "-"   # available but not treating
                else:
                    s_val = " "   # doctor unavailable
            else:
                s_val = str(patient)  # treating patient

            formatted_doctor_schedule.append(left_pad_string(s_val, padding))

        print("doctor:", j, " ".join(formatted_doctor_schedule))

def expand_schedule(Y_values, doctor, T, treat, patient_diseases):
    timeline = [-1 for _ in T]
    for (i, d, t) in Y_values:
        length = treat[doctor][patient_diseases[i]]
        for tt in range(t, t + length):
            if tt < len(T):
                timeline[tt] = i
    return timeline


def create_schedule_from_Z(Z, S, J, T, treat, patient_diseases):
    schedule = []
    for j in J:
        chosen_s = None
        for s in S[j]:
            if Z[j, s].X > 0.5:
                chosen_s = s
                break

        if chosen_s is None:
            doctor_schedule = [-1 for _ in T]
        else:
            _, Y_values = S[j][chosen_s]
            doctor_schedule = expand_schedule(Y_values, j, T, treat, patient_diseases)

        schedule.append(doctor_schedule)
    return schedule


def print_schedule_from_Z(schedule, I, J, T, doctor_times):
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))

    for j, doctor_schedule in zip(J, schedule):
        formatted = []
        for t, patient in enumerate(doctor_schedule):
            if patient == -1:
                s_val = "-"
            else:
                s_val = str(patient)
            if not doctor_times[j][t]:
                s_val = " "
            formatted.append(left_pad_string(s_val, padding))

        print("doctor:", j, " ".join(formatted))
