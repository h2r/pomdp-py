import random
from pomdp_problems.tiger.tiger_problem import TigerState, TigerProblem
import pomdp_py
from pomdp_py.utils.debugging import TreeDebugger

def test_tree_debugger_tiger():

    init_true_state = random.choice([TigerState("tiger-left"),
                                     TigerState("tiger-right")])
    init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                      TigerState("tiger-right"): 0.5})
    tiger_problem = TigerProblem(0.15,  # observation noise
                                 init_true_state, init_belief)

    tiger_problem.agent.set_belief(init_belief, prior=True)
    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                           num_sims=4096, exploration_const=200,
                           rollout_policy=tiger_problem.agent.policy_model)

    pouct.plan(tiger_problem.agent)
    dd = TreeDebugger(tiger_problem.agent.tree)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    test_tree_debugger_tiger()
