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

    # Print BEFORE presolve info (once)
    if "before_presolve_info" in results:
        presolve = results["before_presolve_info"]
        print("="*60)
        print("Model setup / BEFORE presolve info:")
        print("-"*60)
        print(f"Number of variables:       {presolve['num_variables']}")
        print(f"Number of constraints:     {presolve['num_constraints']}")
        print(f"Number of nonzeros:        {presolve['num_nonzeros']}")
        print(f"Setup time (s):            {presolve['setup_time_seconds']:.4f}")
        print("="*60 + "\n")

    # Print objective-specific stats
    for obj_name, data in results.items():
        if obj_name == "before_presolve_info":
            continue  # already printed

        if not "stats" in data:
            continue

        stats = data["stats"]
        schedule = data["schedule"]

        print("\n" + "="*60)
        print(f"Objective: {stats['objective']}")
        print("-"*60)

        # Objective stats
        print(f"Objective value:         {stats['objective_value']:.2f}" if stats["objective_value"] is not None else "Objective value: None")
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

        # Print AFTER presolve info for this objective
        if "after_presolve_info" in data:
            ap = data["after_presolve_info"]
            print("\nAFTER presolve info (for this objective):")
            print(f"Number of columns removed:   {ap['columns_removed']}")
            print(f"Number of rows removed:     {ap['rows_removed']}")
            print(f"Number of variables:     {ap['num_variables']}")
            print(f"Number of constraints:   {ap['num_constraints']}")
            print(f"Elapsed time (s):        {ap['run_time_seconds']:.4f}")
            print("-"*60)

        # Schedule output
        print_schedule(schedule)
        print("="*60 + "\n")


if __name__ == "__main__":

    data_name = "OUTPUT_data_seed10_I100_J10_K4_T20"

    with open(f"{data_name}.json", "r") as f:
        all_model_results = json.load(f)

    print_results(all_model_results)
