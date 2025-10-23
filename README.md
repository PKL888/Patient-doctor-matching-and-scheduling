# Patient-doctor-matching-and-scheduling

## Project structure
Master file (*main.py*) allows the user to create random data instances and run different models on them.

## Folders

- **Data** –     stores generated input data for different problem instances.

- **Outputs** – contains model results, logs, and generated figures.

- **Slides** – includes both the PDF and LaTeX source files for presentation slides.

- **Source code** – all model implementations, data generation scripts, and plotting utilities are contained here.

    - utils/ – data generation, result logging, performance profiling, and summary table scripts.

    - compact/ – smaller-scale model formulations, including feasibility checks, compatible time generation, and doctor availability models.

    - huge/ – large-scale formulations and column generation approaches:

        - naïve column generation

        - subset column generation (default, supports multi-processing)

        - huge formulation

        - fragment column generation and fragment huge formulation

    - pareto/ – ε-constraint algorithm and associated plotting tools for Pareto analysis.

    - benders/ – experimental Benders decomposition model (bonus exploratory work).
