import gurobipy as gp
from data_gen import *
import random
import json

with open("data_seed10_I100_J10_K4_T20.json", "r") as f:
    data = json.load(f)



# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

START = 0
DURATION = 1

I_k = [[i for i in I if patient_diseases[i] == k] for k in K]
J_k = [[j for j in J if qualified[j][k]] for k in K]

diseases_doctor_qualified_for = {j: [k for k in K if qualified[j][k]] for j in J}

# diseases_doctor_can_treat_at_time = {(j,t): [k for k in diseases_doctor_qualified_for[j] if t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] + 1]]
#                                      for j in J for t in T}

# patients_with_disease_that_can_be_treated_at_time = \
# {(k,t):
#     [i for i in I_k[k] if t in T[patient_available[i][START]: patient_available[i][START] + patient_available[i][DURATION]]] 
# for k in K for t in T}

compatible_times = {(i,j):
                    T[max(patient_available[i][START], doctor_available[j][START]):
      (max(0, min(patient_available[i][START] + patient_available[i][DURATION], doctor_available[j][START] + doctor_available[j][DURATION]) - treat[j][k] + 1))]
      for k in K for i in I_k[k] for j in J_k[k]}

# Create a binary list for doctor availability per doctor per time period
doctor_times = []
for d in doctor_available:
    time_list = []
    binary_list = []
    for j in range(d[0], d[0]+d[1]):
        time_list.append(j)
    for i in range(len(T)):
        if i in time_list:
            binary_list.append(1)
        else:
            binary_list.append(0)
    doctor_times.append(binary_list)

# Create a binary list for patient availability per patient per time period 
patient_times = []
for p in patient_available:
    time_list = []
    binary_list = []
    for j in range(p[0], p[0]+p[1]):
        time_list.append(j)
    for i in range(len(T)):
        if i in time_list:
            binary_list.append(1)
        else:
            binary_list.append(0)
    patient_times.append(binary_list)

# generate possible doctor schedules:
# PossibleSchedules = []

# [0,0,0,0,0,5,0,0,0,1,0,0,4,0,0,0]
# [0,0,0,0,0,5,5,5,5,1,1,1,4,4,4,0]

# class Appointment:
#     def __init__(self, doctor, patient, time):
#         self.doctor = doctor
#         self.patient = patient
#         self.disease = diseases_doctor_qualified_for[doctor]

# class DoctorSchedule:
#     def __init__(self, doctor):
#         self.doctor = doctor
#         self.diseases = diseases_doctor_qualified_for[doctor]
#         self.schedule = [-1 for t in T]
    
#     def check_can_add_appointment(self, patient, time):
#         disease = patient_diseases[patient]
    
#     def check_time_for_appointment(self)

#     def add_appointment(self,)

# possible_appointments = 
            


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

SHEDULE_SIZE  = 3

S = dict()
print("starting to gen doctor schedules")
for j in J:
    print("Doctor:", j)
    schedules_with_size_n = dict()
    # Schedules_with_size_n[0] = {}
    schedules_with_size_n[1] = set(frozenset({(i,t,t+treat[j][k])}) for k in diseases_doctor_qualified_for[j] for i in I_k[k] for t in compatible_times[i,j])
    for upto in range(2, SHEDULE_SIZE + 1):
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
    # print(type(schedules_with_size_n[1]))
    # print("size 1", len(schedules_with_size_n[1]), schedules_with_size_n[1])
    # print("size 2", len(schedules_with_size_n[2]), schedules_with_size_n[2])

    DoctorApptSets = set()
    for n in range(1, len(T) + 1):
        DoctorApptSets |= schedules_with_size_n[n]
    # DoctorApptSets = frozenset(frozenset(schedules_with_size_n[n]) for n in range(1,len(T)+1))
    # DoctorApptSets = frozenset(frozenset(schedules_with_size_n[n]) for n in range(1,len(T)+1))

    S[j] = DoctorApptSets
print("Genned doctor schedules")







m = gp.Model("Doctor availability")

"""

# Variables
Y = {(i,j,t):
    m.addVar(vtype=gp.GRB.BINARY)
    for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]
    }

# Constraints
# DoctorsStartOneTreatAtATime = \
# {(j,t):
#  m.addConstr(gp.quicksum(Y.get((i,j,t),0) for k in K for i in I_k[k]) <= 1)
#  for j in J for t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION]]}

DoctorsAreNotOverbooked = \
{(j,t):
 m.addConstr(gp.quicksum(Y[i,j,tt] for k in K if j in J_k[k] for i in I_k[k] 
                         for tt in T[max(0, t - treat[j][k] + 1):t+1] if tt in compatible_times[i, j]) <= 1)
 for j in J for t in T}

PatientsAreAssignedOnlyOnce = \
{i:
 m.addConstr(gp.quicksum(Y[i,j,t] for j in J_k[k] for t in compatible_times[i,j]) <=1)
for k in K for i in I_k[k]}

# compatible_times takes care of the feasible time constraint

# FeasibleTime = \
# {(j,k,t):
#  m.addConstr(treat[j][k] * Y[i,j,t] <= sum(doctor_times[j][tt] * patient_times[i][tt] for tt in range(t, min(t + treat[j][k], len(T)))))
#  for k in K for i in I_k[k] for j in J for t in T}

# compatible_times also takes care of the doctor qualification to treat different diseases

# DoctorsQualified = \
# {(i,j,k,t):
# m.addConstr(Y[i,j,t] <= qualified[j][k])
# for k in K for i in I_k[k] for j in J for t in T}

def left_pad_string(s, length):
    if len(s) >= length:
        return s
    
    return " " * (length - len(s)) + s 

def create_schedule(Ys):
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

def print_stats(Ys):
    numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
    doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
    doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]
    print("Stats: -----------------------------------")
    print("Number of patients allocated:", round(sum(Ys[i,j,t] for i in I for j in J for t in T)))
    print("Patient satisfaction with doctor and time:", round(sum(Ys[i,j,t] * ((numberAvailableDoctors[i] - allocate_rank[i][j] + 1)/numberAvailableDoctors[i] + ((patient_available[i][1]) + 1 - patient_time_prefs[i][t])/patient_available[i][1]) for i in I for j in J for t in T)))
    print("Doctor satisfaction with diseases:", round(sum((doctor_disease_rank_scores[j][k]) * Ys[i,j,t] for k in K for i in I_k[k] for j in J for t in T)))
    print("Appointments per doctor:", round(sum(Ys[i,j,t] for i in I for j in J for t in T))/len(J))

def print_schedule(schedule):
    padding = len(str(len(I)))
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in T]))
    for j in J:
        formatted_doctor_schedule = [(patient >= 0) * str(patient) + (patient < 0) * " " + "-" * (1 - doctor_times[j][t]) for t, patient in enumerate(schedule[j])]
        padded_doctor_shedule = [left_pad_string(s, padding) for s in formatted_doctor_schedule]
        print("doctor:", j, " ".join(padded_doctor_shedule))

def optimise_and_print_schedule():
    m.optimize()
    Yvals = {key: Y[key].x for key in Y}
    Ys = {(i,j,t): Yvals.get((i,j,t), 0) for i in I for j in J for t in T}

    schedule = create_schedule(Ys)
    print_stats(Ys)
    print_schedule(schedule)

# m.setParam("OutputFlag", 0)

# Objective 1: Max. number of matches
# print("Objective 1: Max. number of matches")

# m.setObjective(gp.quicksum(Y[i,j,t] for i in I for j in J for t in T), gp.GRB.MAXIMIZE)
# optimise_and_print_schedule()

# Objective 2: Max. patient satisfaction
print("Objective 2: Max. patient satisfaction")

numberAvailableDoctors = [sum(allocate_rank[i][jj] != M1 for jj in J) for i in I]
patientDoctorScore = [[(numberAvailableDoctors[i] - allocate_rank[i][j] + 1) / numberAvailableDoctors[i] for j in J] for i in I]
patientTimeScore = [[(patient_available[i][1] + 1 - patient_time_prefs[i][t]) / patient_available[i][1] for t in T] for i in I]

m.setObjective(gp.quicksum(Y[i,j,t] * 
                           (
                                patientDoctorScore[i][j] 
                                + 
                                sum(patientTimeScore[i][t:min(t + treat[j][k], len(T))]) / treat[j][k]
                            )
                           for k in K for i in I_k[k] for j in J_k[k] for t in compatible_times[i,j]), gp.GRB.MAXIMIZE)
optimise_and_print_schedule()

# Objective 3: Max. doctor satisfaction
# print("Objective 3: Max. doctor satisfaction")

# doctor_num_diseases_can_treat = [sum(qualified[j]) for j in J]
# doctor_disease_rank_scores = [[qualified[j][k] * (doctor_num_diseases_can_treat[j] - doctor_rank[j][k] + 1)/doctor_num_diseases_can_treat[j] + (1 - qualified[j][k]) * -M1 for k in K] for j in J]

# m.setObjective(gp.quicksum((doctor_disease_rank_scores[j][k]) * Y[i,j,t] for k in K for i in I_k[k] for j in J for t in T), gp.GRB.MAXIMIZE)
# optimise_and_print_schedule()

"""

