from doctor_available_IP import *

import re

# function for taking from the log and checking against original number of vars and constraints
def parse_presolve_log(logfile="gurobi_presolve.log"):
    presolve_info = {
        "rows_removed": 0,
        "columns_removed": 0,
        "nonzeros_removed": 0
    }
    
    with open(logfile, "r") as f:
        for line in f:
            # match "Presolve removed X rows and Y columns"
            match = re.search(r"Presolve removed (\d+) rows? and (\d+) columns?", line)
            if match:
                # calculate the variables and constraints from before presolve minus
                # the rows and cols removed in presolve
                presolve_info["num_variables"] = m.NumVars - int(match.group(2))
                presolve_info["num_constraints"] = m.NumConstrs - int(match.group(1))

    return presolve_info


