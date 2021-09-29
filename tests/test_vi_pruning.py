import sys
import pomdp_py
import subprocess
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdp_file
from pomdp_py.utils.interfaces.solvers import vi_pruning
import os
import io

description="testing vi_pruning (pomdp-solve)"

def test_vi_pruning(pomdp_solve_path, return_policy_graph=True):
    print("[testing] test_vi_pruning")
    tiger = make_tiger()

    # Building a policy graph
    print("[testing] solving the tiger problem...")
    policy = vi_pruning(tiger.agent, pomdp_solve_path, discount_factor=0.95,
                        options=["-horizon", "100"],
                        remove_generated_files=True,
                        return_policy_graph=return_policy_graph)

    assert str(policy.plan(tiger.agent)) == "listen",\
        "Bad solution. Test failed."

    # Plan with the graph for several steps. So we should get high rewards
    # eventually in the tiger domain.
    got_high_reward = False
    for step in range(10):
        true_state = tiger.env.state
        action = policy.plan(tiger.agent)
        observation = tiger.agent.observation_model.sample(true_state, action)
        reward = tiger.env.reward_model.sample(true_state, action, None)
        print("[testing] simulating computed policy graph"\
              "(step=%d, action=%s, observation=%s, reward=%d)" % (step, action, observation, reward))
        # No belief update needed. Just update the policy graph
        if return_policy_graph:
            # We use policy graph. No belief update is needed. Just update the policy.
            policy.update(tiger.agent, action, observation)
        else:
            # belief update is needed
            new_belief = pomdp_py.update_histogram_belief(tiger.agent.cur_belief,
                                                          action, observation,
                                                          tiger.agent.observation_model,
                                                          tiger.agent.transition_model)
            tiger.agent.set_belief(new_belief)

        assert reward == -1 or reward == 10, "Reward is negative. Failed."
        if reward == 10:
            got_high_reward = True
    assert got_high_reward, "Should have gotten high reward. Failed."
    print("Pass.")

def _check_pomdp_solve():
    pomdp_solve_path = os.getenv("POMDP_SOLVE_PATH")
    if pomdp_solve_path is None or not os.path.exists(pomdp_solve_path):
        raise FileNotFoundError("To run this test, download pomdp-solve from "
                                "https://www.pomdp.org/code/. Then, follow the "
                                "instructions on this web page to compile this software. "
                                "Finally, set the environment variable POMDP_SOLVE_PATH "
                                "to be the path to the pomdp-solve binary file "
                                "generated after compilation, likely located at "
                                "/path/to/pomdp-solve-<version>/src/pomdp-solve ")
    return pomdp_solve_path

def run():
    pomdp_solve_path = _check_pomdp_solve()
    test_vi_pruning(pomdp_solve_path, return_policy_graph=False)

if __name__ == "__main__":
    run()
