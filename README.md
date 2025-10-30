# Patient-doctor-matching-and-scheduling

This implements various interger programs using gurobi to solve a doctor-patient schedule problem where availabilities are heterogeneous, patients rank doctors and times, and doctors rank diseases.

## Project structure
Master file (*main.py*) allows the user to create random data instances and run different models on them.

## Folders

- **Data** –     stores generated input data for different problem instances.

- **Outputs** – contains model results, logs, and generated figures.

- **Slides** – includes both the PDF and LaTeX source files for presentation slides.

- **Source code** – all model implementations, data generation scripts, and plotting utilities are contained here.

    - utils/ – data generation, result logging, performance profiling, and summary table scripts.

    - compact/ – direct integer programs based on *feasibility* checks, *compatible times* between patients and doctors, and *doctor availability* models.

    - huge/ – huge formulations using apriori column generation where variables represent groups of appointments or whole doctor schedules:

        - subset column generation (default, supports multi-processing)

        - huge formulation

        - fragment column generation and fragment huge formulation

    - pareto/ – ε-constraint algorithm and associated plotting tools for Pareto analysis.

    - benders/ – experimental Benders decomposition model (bonus exploratory work).
