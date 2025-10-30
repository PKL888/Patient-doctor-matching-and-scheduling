import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from compact.compatible_times import make_compatible_times_model 
from compact.doctor_available import make_doctor_available_model 
from compact.feasibility import make_feasibility_model 
from huge.cg_fragments_formulation import make_huge_frag_model
from huge.cg_huge import make_huge_model
from pareto.epsilon_10_instances import epsilon_runs
from pareto.frontier import *
from utils.data_instance import DataInstance
from utils.graph_1000_seed_model_results import plot_model_files
from utils.print_model_results import summarize_pareto_slack_results

FEASIBILITY = 1
COMPATIBLE_TIMES = 2
DOCTOR_AVAILABLE = 3
SUBSET_COLUMN_GEN = 8
FRAGMENT_COLUMN_GEN = 9

model_names = {
    FEASIBILITY: "feasibility",
    COMPATIBLE_TIMES: "compatible_times",
    DOCTOR_AVAILABLE: "doctor_available",
    SUBSET_COLUMN_GEN: "subset_column_gen",
    FRAGMENT_COLUMN_GEN: "fragment_column_gen"
}

NUM_APPOINTMENTS = 1
PAT_SAT = 2
DOC_SAT = 3
PARETO = 5

def make_model(model:int, data: dict, d: DataInstance):
    Y = None
    Z = None
    S = None
    F = None
    W = None

    if model in [FEASIBILITY, COMPATIBLE_TIMES, DOCTOR_AVAILABLE]:
        if model == FEASIBILITY:
            m, Y, [objective_0, objective_1, objective_2], time = make_feasibility_model(d, data)
        elif model == COMPATIBLE_TIMES:
            m, Y, [objective_0, objective_1, objective_2], time = make_compatible_times_model(data)
        elif model == DOCTOR_AVAILABLE:
            m, Y, [objective_0, objective_1, objective_2], time = make_doctor_available_model(data)
    
    elif model == SUBSET_COLUMN_GEN:
        m, Z, [objective_0, objective_1, objective_2], time, S = make_huge_model(d, d.seed, len(d.I), len(d.J), len(d.K), len(d.T))
    
    elif model == FRAGMENT_COLUMN_GEN:
        max_frag_length = int(input("Maximum fragment length (at least 2, 5 and up is very slow):    "))
        m, W, [objective_0, objective_1, objective_2], time, F = make_huge_frag_model(d, max_frag_length)
    
    print(f"[INFO] Generating the model has taken {round(time, 4)} seconds")
    
    return m, [objective_0, objective_1, objective_2], Y, Z, W, S, F

def user_problem():
    print("Welcome to our interactive modelling tool.")
    print("\nSelect problem size:")
    print("1: Tiny   (patients: 10,  doctors: 2,  diseases: 2, time periods: 10)")
    print("2: Small  (patients: 50,  doctors: 5,  diseases: 3, time periods: 15)")
    print("3: Medium (patients: 100, doctors: 10, diseases: 4, time periods: 20)")

    size_choice = input("Enter 1, 2, or 3:    ")
    while size_choice not in ["1", "2", "3"]:
        print(f"{size_choice} is not a valid choice.")
        size_choice = input("Enter 1, 2, or 3: ")

    if size_choice == "1":
        problem_size = {"patients": 10, "doctors": 2, "diseases": 2, "time periods": 10}
    elif size_choice == "2":
        problem_size = {"patients": 50, "doctors": 5, "diseases": 3, "time periods": 15}
    else:
        problem_size = {"patients": 100, "doctors": 10, "diseases": 4, "time periods": 20}

    customise = input("\nWould you like to customise these numbers? (y/n):    ").lower()
    if customise == "y":
        for key in problem_size:
            val = input(f"Enter number of {key} (default {problem_size[key]}):    ")
            if val.isdigit():
                problem_size[key] = int(val)

    print(f"Selected problem size: {problem_size}")

    # Ask the user for number of random instances
    seed_choice = input("\nHow many random instances would you like to generate? Enter number:    ")

    while not seed_choice.isdigit() or int(seed_choice) < 1:
        print(f"{seed_choice} is not a valid positive number.")
        seed_choice = input("Enter the number of random instances (positive integer):    ")

    num_instances = int(seed_choice)

    # Generate seeds
    if num_instances == 1:
        seeds = [0]
    else:
        seeds = list(range(num_instances))

    print(f"Seeds to be used: {seeds}")
    return seeds, problem_size

def user_model():
    model = int(input(f"\nPlease choose model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available, {SUBSET_COLUMN_GEN}: schedule_column_gen, {FRAGMENT_COLUMN_GEN}: fragment_column_gen]:    "))
    while model not in [FEASIBILITY,COMPATIBLE_TIMES,DOCTOR_AVAILABLE,SUBSET_COLUMN_GEN,FRAGMENT_COLUMN_GEN]:
        print(f"{model} is not a valid model.")
        model = int(input(f"Please choose model: [{FEASIBILITY}: feasibility, {COMPATIBLE_TIMES}: compatible_times, {DOCTOR_AVAILABLE}: doctor_available, {SUBSET_COLUMN_GEN}: schedule_column_gen, {FRAGMENT_COLUMN_GEN}: fragment_column_gen]:    "))
    return model

def user_objective():
    obj = int(input(f"Please choose objective: [{NUM_APPOINTMENTS}: number of appointments, {PAT_SAT}: patient satisfaction, {DOC_SAT}: doctor satisfaction, {PARETO}: pareto frontier]:    "))
    while obj not in [NUM_APPOINTMENTS,PAT_SAT,DOC_SAT,PARETO]:
        print(f"{obj} is not a valid objective.")
        obj = int(input(f"Please choose objective: [{NUM_APPOINTMENTS}: number of appointments, {PAT_SAT}: patient satisfaction, {DOC_SAT}: doctor satisfaction, {PARETO}: pareto frontier]:    "))
    return obj

def set_objective(m: gp.Model, obj: int, objectives):
    [objective_0, objective_1, objective_2] = objectives

    if (obj not in [NUM_APPOINTMENTS, PAT_SAT, DOC_SAT]):
        print(f"{obj} is not a valid objective")
        return
    
    if (obj == NUM_APPOINTMENTS):
        print("set objective as num appointments", NUM_APPOINTMENTS)
        m.setObjective(objective_0, gp.GRB.MAXIMIZE)
    elif (obj == PAT_SAT):
        print("set objective as pat sat", PAT_SAT)
        m.setObjective(objective_1, gp.GRB.MAXIMIZE)
    elif (obj == DOC_SAT):
        print("set objective as doc_sat", DOC_SAT)
        m.setObjective(objective_2, gp.GRB.MAXIMIZE)

def user_plot_performance():
    plot_performance = input("Would you like to plot the performance profiles for the compact models or column generation algorithms? (y/n):       ").lower()
    if plot_performance == "y":
        plot = int(input("Which performance profiles would you like to plot?: 1: Compact models (single-objectives), 2: Column generation, 3: Compact IP (multi-objective), 4: None       "))
        
        if (plot == 1):
            model_files = {
                "Feasibility": "outputs/results/F_all_1000_seeds_model_results.pkl",
                "Compatible times": "outputs/results/CT_all_1000_seeds_model_results.pkl",
                "Doctor available": "outputs/results/DA_all_1000_seeds_model_results.pkl",
            }
            plot_model_files(model_files, 1)
        
        elif (plot == 2):
            model_files = {
                "Singular:": "src/huge/cg_schedules_timed_all_100_seeds_model_results.pkl",
                "Multiproccessing: ": "src/huge/cg_schedules_timed_multiproccessing_all_100_seed_model_results.pkl"
            }
            plot_model_files(model_files, 2)
        
        elif (plot == 3):
            model_files = {}
            plot_model_files(model_files, 3)

def user_table(model, model_type):
    pareto_table = input("\nWould you like to save the multi-objective model output for 10 instances in a table? (y/n):    ").lower()
    if pareto_table == "y":
        pareto_table_size = int(input("Which model size would you like to save? 1: Small, 2: Medium, 3: Large, 4: None     "))
        make_epsilon_summary_table(pareto_table_size, model, model_type)
        
        if pareto_table != 4:
            plt.savefig("outputs/images/epsilon_summary.png", dpi=300)
            plt.show()

def user_plot_frontier(seeds, all_data, model):
    pareto_plot = input("Would you like to plot the Pareto frontier? (y/n):    ").lower()
    if pareto_plot == "y":
        for seed in seeds:
            data = all_data[f"seed_{seed}"]
            d = DataInstance(data)

            path = "outputs/results"
            filename = (f"{path}/pareto_{model_names[model]}_seed{d.seed}_I{len(d.I)}_J{len(d.J)}_K{len(d.K)}_T{len(d.T)}.pkl")
            with open(filename, "rb") as f:
                pareto = pickle.load(f)

            output_path = "outputs/graphs"
            plot_pareto_2d(pareto[pareto_slack], pareto[dom_slack], save_path=output_path)
            plot_pareto_3d(pareto[pareto_slack], save_path=output_path)

def make_epsilon_summary_table(pareto_table: int, model, model_type=-1):
    if (pareto_table == 1):
        patient_sizes = [30, 40, 50]
        doctor_sizes = [3, 5, 7]

    elif (pareto_table == 2):
        patient_sizes = [100, 110, 120]
        doctor_sizes = [8, 10, 12]

    elif (pareto_table == 3):
        patient_sizes = [230, 240, 250]
        doctor_sizes = [15, 18, 25]

    if (pareto_table != 4):
        # Seeds 1 through 10
        seeds = range(1, 11)

        summary_rows = []
        max_frag_length = 0    
        if model_type == FRAGMENT_COLUMN_GEN and not max_frag_length:
            max_frag_length = int(input("Maximum fragment length (at least 2, 5 and up is very slow):    "))

        for i in patient_sizes:
            for j in doctor_sizes:
                epsilon_problem_size = {
                    "patients": i,
                    "doctors":  j,
                    "diseases": 3,
                    "time periods": 20
                }
        
                epsilon_runs(model, epsilon_problem_size, seeds, frag_length=max_frag_length)

                summary = summarize_pareto_slack_results(seeds, i, j, 3, 20, model)
                summary_rows.append(summary)
        
        
        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.set_index(["Patients", "Doctors"])
        print(df_summary)
        df_summary.to_csv(f"outputs/images/epsilon_summary{pareto_table=},{model=}.csv")

        df_plot = df_summary.reset_index()

        fig, ax = plt.subplots(figsize=(8, len(df_plot)*0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=df_plot.values,
            colLabels=df_plot.columns,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df_plot.columns))))

        plt.tight_layout()
        plt.savefig(f"outputs/images/epsilon_summary{pareto_table=},{model=}.png", dpi=300)
        plt.show()
