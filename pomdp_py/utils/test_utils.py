import pomdp_py
from pomdp_problems.tiger.tiger_problem import TigerProblem
from pomdp_problems.tiger.tiger_problem import TigerState
import os
import glob

def remove_files(pattern):
    file_list = glob.glob(pattern)
    for file_path in file_list:
        os.remove(file_path)

def make_tiger(noise=0.15, init_state="tiger-left", init_belief=[0.5, 0.5]):
    """Convenient function to quickly build a tiger domain.
    Useful for testing"""
    tiger = TigerProblem(noise, TigerState(init_state),
                         pomdp_py.Histogram({TigerState("tiger-left"): init_belief[0],
                                             TigerState("tiger-right"): init_belief[1]}))
    return tiger
