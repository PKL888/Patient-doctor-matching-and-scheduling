import json

def left_pad_string(s, length):
    """Helper to align output columns."""
    if len(s) >= length:
        return s
    return " " * (length - len(s)) + s 

def print_schedule(schedule):
    """Pretty-print the doctor schedules."""
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
    """Print presolve info, stats, and schedule for each objective."""

    # Print presolve info if available
    if "presolve_info" in results:
        presolve = results["presolve_info"]
        print("="*60)
        print("Presolve / Model setup info:")
        print("-"*60)
        print(f"Number of variables:       {presolve['num_variables']}")
        print(f"Number of constraints:     {presolve['num_constraints']}")
        print(f"Number of nonzeros:        {presolve['num_nonzeros']}")
        print(f"Setup / presolve time (s): {presolve['setup_time_seconds']:.4f}")
        print("="*60 + "\n")

    # Print objective-specific stats
    for obj_name, data in results.items():
        if obj_name == "presolve_info":
            continue  # already printed

        stats = data["stats"]
        schedule = data["schedule"]

        print("\n" + "="*60)
        print(f"Objective: {stats['objective']}")
        print("-"*60)
        print(f"Objective value:         {stats['objective_value']:.2f}" if stats["objective_value"] is not None else "Objective value:   None")
        print(f"Patients allocated:      {stats['num_patients_allocated']}")
        print(f"Patient satisfaction:    {stats['patient_satisfaction']}")
        print(f"Doctor satisfaction:     {stats['doctor_satisfaction']}")
        print(f"Appointments per doctor: {stats['appointments_per_doctor']:.2f}")

        # Solver stats
        print("\nSolver stats:")
        print(f"Runtime (s):             {stats['runtime']:.2f}")
        if stats["mip_gap"] is not None:
            print(f"MIP Gap:                 {stats['mip_gap']:.4f}")
        print(f"Nodes explored:          {stats['nodes']}")
        print(f"Iterations:              {stats['iterations']}")
        print(f"Solutions found:         {stats['solutions_found']}")

        # Schedule output
        print_schedule(schedule)
        print("="*60 + "\n")


if __name__ == "__main__":
    with open("all_model_results.json", "r") as f:
        all_model_results = json.load(f)

    print_results(all_model_results)
