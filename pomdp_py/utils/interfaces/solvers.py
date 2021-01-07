"""
Provides function calls to use external solvers,
given a POMDP defined using pomdp_py interfaces.

Currently, we interface with:
* pomdp-solve: http://www.pomdp.org/code/index.html
* SARSOP

We plan to interface with:
* POMDP.jl
"""
import pomdp_py
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, parse_pomdp_solve_output
import subprocess
import os

def vi_pruning(agent, pomdp_solve_path,
               discount_factor=0.95,
               options=[]):
    """
    Value Iteration with pruning, using the software pomdp-solve
    https://www.pomdp.org/code/ developed by Anthony R. Cassandra.

    Args:
        agent (pomdp_py.Agent): The agent that contains the POMDP definition
        pomdp_solve_path (str): Path to the `pomdp_solve` binary generated after
            compiling the pomdp-solve library.
        options (list): Additional options to pass in to the command line interface.
             The options should be a list of strings, such as ["-stop_criteria", "weak", ...]
             Some useful options are:
                 -horizon <int>
                 -time_limit <int>
    """
    try:
        all_states = list(agent.all_states)
        all_actions = list(agent.all_actions)
        all_observations = list(agent.all_observations)
    except NotImplementedError:
        print("S, A, O must be enumerable for a given agent to convert to .pomdp format")

    pomdp_path = "./temp-pomdp.pomdp"
    to_pomdp_file(agent, pomdp_path, discount_factor=discount_factor)
    proc = subprocess.Popen([pomdp_solve_path,
                             "-pomdp", pomdp_path,
                             "-o", "temp-pomdp"] + options)
    proc.wait()

    # Read the value and policy graph files
    alpha_path = "temp-pomdp.alpha"
    pg_path = "temp-pomdp.pg"
    alphas, pg = parse_pomdp_solve_output(alpha_path, pg_path)
    policy_graph = PolicyGraph.construct(alphas, pg,
                                         all_states, all_actions, all_observations)

    # Remove temporary files
    os.remove(pomdp_path)
    os.remove(alpha_path)
    os.remove(pg_path)
    return policy_graph
