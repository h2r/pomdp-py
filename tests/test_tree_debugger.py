import random
from pomdp_problems.tiger import TigerProblem, test_planner
import pomdp_py
from pomdp_py.utils.debugging import TreeDebugger

description="testing pomdp_py.utils.TreeDebugger"

def test_tree_debugger_tiger(debug_tree=False):
    tiger_problem = TigerProblem.create("tiger-left", 0.5, 0.15)
    pouct = pomdp_py.POUCT(max_depth=4, discount_factor=0.95,
                           num_sims=4096, exploration_const=200,
                           rollout_policy=tiger_problem.agent.policy_model)

    pouct.plan(tiger_problem.agent)
    dd = TreeDebugger(tiger_problem.agent.tree)

    # The number of VNodes equals to the sum of VNodes per layer
    assert dd.nv == sum([len(dd.l(i)) for i in range(dd.nl)])

    # The total number of nodes equal to the number of VNodes plus QNodes
    assert dd.nn == dd.nv + dd.nq

    # Test example usage
    dd.mark(dd.path(dd.layer(2)[0]))
    print("Printing tree up to depth 1")
    dd.p(1)

    # There exists a path from the root to nodes in the tree
    for i in range(dd.nl):
        n = dd.l(i)[0]
        path = dd.path_to(n)
        assert path is not None

    test_planner(tiger_problem, pouct, nsteps=3, debug_tree=debug_tree)

def run(verbose=False, debug_tree=False):
    test_tree_debugger_tiger(debug_tree=debug_tree)


if __name__ == "__main__":
    run(debug_tree=True)
