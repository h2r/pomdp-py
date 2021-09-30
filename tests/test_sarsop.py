import sys
import pomdp_py
import subprocess
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdp_file
from pomdp_py.utils.interfaces.solvers import sarsop
import os
import io

description="testing sarsop"

def test_sarsop(pomdpsol_path):
    print("[testing] test_sarsop")
    tiger = make_tiger()

    # Building a policy graph
    print("[testing] solving the tiger problem...")
    policy = sarsop(tiger.agent, pomdpsol_path, discount_factor=0.95,
                    timeout=10, memory=20, precision=0.000001,
                    remove_generated_files=True,
                    logfile="test_sarsop.log")

    assert str(policy.plan(tiger.agent)) == "listen",\
        "Bad solution. Test failed."

    assert os.path.exists("test_sarsop.log")
    os.remove("test_sarsop.log")

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

def _check_pomdpsol():
    pomdpsol_path = os.getenv("POMDPSOL_PATH")
    if pomdpsol_path is None or not os.path.exists(pomdpsol_path):
        raise FileNotFoundError("To run this test, download sarsop from"
                                "https://github.com/AdaCompNUS/sarsop. Then, follow the "
                                "instructions on this web page to compile this software. "
                                "Finally, set the environment variable POMDPSOL_PATH "
                                "to be the path to the pomdpsol binary file "
                                "generated after compilation, likely located at "
                                "/path/to/sarsop/src/pomdpsol")
    return pomdpsol_path

def run():
    pomdpsol_path = _check_pomdpsol()
    test_sarsop(pomdpsol_path)


if __name__ == "__main__":
    run()
