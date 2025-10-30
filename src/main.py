from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from outputs.results import *
from pareto.epsilon import make_pareto_frontier
from utils.data_gen import get_data
from utils.data_instance import DataInstance
from utils.logging_results import optimise_and_print_schedule
from utils.user_input import *
from src.huge import *

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

if __name__ == '__main__':
    seeds, problem_size = user_problem()
    model, model_type = user_model()
    obj = user_objective()

    all_data = get_data(problem_size, seeds)
    for seed in seeds:
        data = all_data[f"seed_{seed}"]
        d = DataInstance(data)

        m, [objective_0, objective_1, objective_2], Y, Z, W, S, F = make_model(model, data, d)

        if obj == PARETO:
            m.setParam("OutputFlag", 0)
            make_pareto_frontier(data, m, Y, Z, W, S, F, d.I, d.J, d.K, d.T, [objective_0, objective_1, objective_2], model, d, dense=False)
            
        else:
            set_objective(m, obj, [objective_0, objective_1, objective_2])

            m.setParam("OutputFlag", 1)
            optimise_and_print_schedule(model_type, model, m, Y, Z, W, S, F, data, d, True)
                
    user_plot_performance()

    if obj == PARETO:
        user_table(model, model_type)
        user_plot_frontier(seeds, all_data, model)

    print("\nModelling complete :)")
    
    # for size in [1,2,3]:
    #     for model in [DOCTOR_AVAILABLE, FEASIBILITY, COMPATIBLE_TIMES]:
    #         make_epsilon_summary_table(size, model)
 
    # was used to run the epsilon model on 3 basic models over 100 instances
    # model_names = {
    #     "1": FEASIBILITY,
    #     "2": COMPATIBLE_TIMES,
    #     "3": DOCTOR_AVAILABLE       
    # }

    # new_seeds = range(1, 101)
    # new_problem_size = {
    #     "patients": 50,
    #     "doctors":  5,
    #     "diseases": 4,
    #     "time periods": 20
    # }
    # for model in model_names.values():
    #     epsilon_runs(model, new_problem_size, new_seeds, dense=False)
                