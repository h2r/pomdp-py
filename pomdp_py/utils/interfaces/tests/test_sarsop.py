import sys
import pomdp_py
import subprocess
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdp_file
from pomdp_py.utils.interfaces.solvers import sarsop
import os
import io


def test_sarsop(pomdpsol_path):
    print("[testing] test_sarsop")
    tiger = make_tiger()

    # Building a policy graph
    print("[testing] solving the tiger problem...")
    policy = sarsop(tiger.agent, pomdpsol_path, discount_factor=0.95,
                    timeout=10, memory=20, precision=0.000001,
                    remove_generated_files=True)

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
        print("[testing] running computed policy graph"\
              "(step=%d, action=%s, observation=%s, reward=%d)" % (step, action, observation, reward))

        # belief update
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To run test, do: %s <pomdp-solver-path>" % sys.argv[0])
        print("Download pomdp-solve from https://www.pomdp.org/code/")
        exit(1)
    solver_path = sys.argv[1]
    test_sarsop(solver_path)
