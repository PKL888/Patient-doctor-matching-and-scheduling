import json
# import to be able to graph
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("all_1000_seeds_model_results.json", "r", encoding="utf-8") as f:
    all_model_results = json.load(f)

# all_model_results is { "max_matches": {seed: {...}}, "patient_satisfaction": {...}, ... }

def extract_runtimes(model_results):
    runtimes = [res["stats"]["runtime"] for res in model_results.values()]
    runtimes = np.array(runtimes)
    runtimes.sort()
    y = np.arange(1, len(runtimes)+1) / len(runtimes) * 100  # percentage
    return runtimes, y

# Extract data
x_max, y_max = extract_runtimes(all_model_results["max_matches"])
x_patient, y_patient = extract_runtimes(all_model_results["patient_satisfaction"])
x_doctor, y_doctor = extract_runtimes(all_model_results["doctor_satisfaction"])

# Plot ECDF (runtime distribution)
plt.figure(figsize=(10,6))
plt.plot(x_max, y_max, label="Max Matches", color="blue", linestyle="-")
plt.plot(x_patient, y_patient, label="Patient Satisfaction", color="green", linestyle="--")
plt.plot(x_doctor, y_doctor, label="Doctor Satisfaction", color="red", linestyle=":")

plt.xscale("log")  # often runtime distributions are on log scale
plt.xlabel("Runtime (s)")
plt.ylabel("Solved Instances (%)")
plt.title("Runtime Distribution of Models")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
