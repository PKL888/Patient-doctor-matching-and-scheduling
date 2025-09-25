import gurobipy as gp
from data_gen import *
from schedule_printing import *
from logging_results import *
import random
import pickle
import json
import time

with open("data_seed10_I100_J10_K4_T20.pkl", "rb") as f:
    data = pickle.load(f)

# put everything in the global namespace
globals().update(data)

I = range(problem_size["patients"])
J = range(problem_size["doctors"])
K = range(problem_size["diseases"])
T = range(problem_size["time periods"])

START = 0
DURATION = 1

# diseases_doctor_can_treat_at_time = {(j,t): [k for k in diseases_doctor_qualified_for[j] if t in T[doctor_available[j][START]:doctor_available[j][START] + doctor_available[j][DURATION] - treat[j][k] + 1]]
#                                      for j in J for t in T}

# patients_with_disease_that_can_be_treated_at_time = \
# {(k,t):
#     [i for i in I_k[k] if t in T[patient_available[i][START]: patient_available[i][START] + patient_available[i][DURATION]]] 
# for k in K for t in T}

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