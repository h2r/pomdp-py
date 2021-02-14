"""
`pomdp_py <https://h2r.github.io/pomdp-py/html/>`_ provides function calls to use external solvers,
given a POMDP defined using pomdp_py interfaces. Currently, we interface with:

* `pomdp-solve <http://www.pomdp.org/code/index.html>`_ by Anthony R. Cassandra
* `SARSOP <https://github.com/AdaCompNUS/sarsop>`_ by NUS

We hope to interface with:

* `POMDP.jl <https://github.com/JuliaPOMDP/POMDPs.jl>`_
* more? Help us if you can!
"""
import pomdp_py
from pomdp_py.utils.interfaces.conversion\
    import to_pomdp_file, PolicyGraph, AlphaVectorPolicy, parse_pomdp_solve_output
import subprocess
import os, sys

def vi_pruning(agent, pomdp_solve_path,
               discount_factor=0.95,
               options=[],
               pomdp_name="temp-pomdp",
               remove_generated_files=False,
               return_policy_graph=False):
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
        pomdp_name (str): The name used to create the .pomdp file.
        remove_generated_files (bool): True if after policy is computed,
            the .pomdp, .alpha, .pg files are removed. Default is False.
        return_policy_graph (bool): True if return the policy as a PolicyGraph.
            By default is False, in which case an AlphaVectorPolicy is returned.

    Returns:
       PolicyGraph or AlphaVectorPolicy: The policy returned by the solver.
    """
    try:
        all_states = list(agent.all_states)
        all_actions = list(agent.all_actions)
        all_observations = list(agent.all_observations)
    except NotImplementedError:
        raise("S, A, O must be enumerable for a given agent to convert to .pomdp format")

    pomdp_path = "./%s.pomdp" % pomdp_name
    to_pomdp_file(agent, pomdp_path, discount_factor=discount_factor)
    proc = subprocess.Popen([pomdp_solve_path,
                             "-pomdp", pomdp_path,
                             "-o", pomdp_name] + list(map(str,options)))
    proc.wait()

    # Read the value and policy graph files
    alpha_path = "%s.alpha" % pomdp_name
    pg_path = "%s.pg" % pomdp_name
    if return_policy_graph:
        policy = PolicyGraph.construct(alpha_path, pg_path,
                                       all_states, all_actions, all_observations)
    else:
        policy = AlphaVectorPolicy.construct(
            alpha_path, all_states, all_actions, solver="pomdp-solve")

    # Remove temporary files
    if remove_generated_files:
        os.remove(pomdp_path)
        os.remove(alpha_path)
        os.remove(pg_path)
    return policy


def sarsop(agent, pomdpsol_path,
           discount_factor=0.95,
           timeout=30, memory=100,
           precision=0.5,
           pomdp_name="temp-pomdp",
           remove_generated_files=False,
           logfile=None):
    """
    SARSOP, using the binary from https://github.com/AdaCompNUS/sarsop
    This is an anytime POMDP planning algorithm

    Args:
        agent (pomdp_py.Agent): The agent that defines the POMDP models
        pomdpsol_path (str): Path to the `pomdpsol` binary
        timeout (int): The time limit (seconds) to run the algorithm until termination
        memory (int): The memory size (mb) to run the algorithm until termination
        precision (float): solver runs until regret is less than `precision`
        pomdp_name (str): Name of the .pomdp file that will be created when solving
        remove_generated_files (bool): Remove created files during solving after finish.
        logfile (str): Path to file to write the log of both stdout and stderr
    Returns:
       AlphaVectorPolicy: The policy returned by the solver.
    """
    try:
        all_states = list(agent.all_states)
        all_actions = list(agent.all_actions)
        all_observations = list(agent.all_observations)
    except NotImplementedError:
        raise("S, A, O must be enumerable for a given agent to convert to .pomdpx format")

    if logfile is None:
        stdout = None
        stderr = None
    else:
        logf = open(logfile, "w")
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT

    pomdp_path = "./%s.pomdp" % pomdp_name
    to_pomdp_file(agent, pomdp_path, discount_factor=discount_factor)
    proc = subprocess.Popen([pomdpsol_path,
                             "--timeout", str(timeout),
                             "--memory", str(memory),
                             "--precision", str(precision),
                             "--output", "%s.policy" % pomdp_name,
                             pomdp_path], stdout=stdout, stderr=stderr)
    if logfile is not None:
        for line in proc.stdout:
            line = line.decode("utf-8")
            sys.stdout.write(line)
            logf.write(line)
    proc.wait()

    policy_path = "%s.policy" % pomdp_name
    policy = AlphaVectorPolicy.construct(policy_path,
                                         all_states, all_actions)

    # Remove temporary files
    if remove_generated_files:
        os.remove(pomdp_path)
        os.remove(policy_path)
    if logfile is not None:
        logf.close()
    return policy
