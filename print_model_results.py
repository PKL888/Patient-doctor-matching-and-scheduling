import json

def left_pad_string(s, length):
    """Helper to align output columns."""
    if len(s) >= length:
        return s
    return " " * (length - len(s)) + s 

def print_schedule(schedule):
    """Pretty-print the doctor schedules."""
    # Find max schedule length (number of time periods)
    T = len(next(iter(schedule.values())))
    padding = len(str(T))

    print("\nSchedule:")
    print("time:     " + " ".join([left_pad_string(str(t), padding) for t in range(T)]))

    for doctor, slots in schedule.items():
        formatted_doctor_schedule = [
            str(patient) if patient >= 0 else " " for patient in slots
        ]
        padded = [left_pad_string(s, padding) for s in formatted_doctor_schedule]
        print(f"{doctor}: " + " ".join(padded))


def print_results(results):
    """Print stats + schedule for each objective."""
    for obj_name, data in results.items():
        stats = data["stats"]
        schedule = data["schedule"]

        print("\n" + "="*60)
        print(f"Objective: {stats['objective']}")
        print("-"*60)
        print(f"Objective value:         {stats['objective_value']:.2f}")
        print(f"Patients allocated:      {stats['num_patients_allocated']}")
        print(f"Patient satisfaction:    {stats['patient_satisfaction']}")
        print(f"Doctor satisfaction:     {stats['doctor_satisfaction']}")
        print(f"Appointments per doctor: {stats['appointments_per_doctor']:.2f}")
        
        print_schedule(schedule)
        print("="*60 + "\n")


if __name__ == "__main__":

    with open("all_model_results.json", "r") as f:
        all_model_results = json.load(f)

    print_results(all_model_results)
