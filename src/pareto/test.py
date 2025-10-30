import pickle
from frontier import *

filename = "outputs/results/pareto_doctor_available_seed1_I30_J3_K4_T20.pkl"
with open(filename, "rb") as f:
    pareto = pickle.load(f)

globals().update(pareto)
print(len(pareto_slack))
print(len(pareto_dense))

print()

print(len(dom_slack))
print(len(dom_dense))

print()

print(round(slack_time,2))
print(round(dense_time,2))

missing = [x for x in pareto_dense if x not in pareto_slack]
for i in missing:
    print(i)

path = "outputs/graphs"
plot_pareto_2d(pareto_slack, dom_slack, save_path=path)
plot_pareto_3d(pareto_slack, save_path=path)
