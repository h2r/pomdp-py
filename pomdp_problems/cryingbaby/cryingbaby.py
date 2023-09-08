"""
Example of defining a small, tabular POMDP using pomdp-py
and solving it with different methods.  It illustrates a different
style of using pomdp-py when the domain is very small.

Refer to documentation:
https://h2r.github.io/pomdp-py/html/examples.external_solvers.html
"""
import pomdp_py
import os

def cryingbaby():
    """This is a POMDP defined in the Algorithms for Decision Making book
    by M. J. Kochenderfer et al. in section F.7"""
    S = ['hungry', 'sated']
    A = ['feed', 'sing', 'ignore']
    Z = ['crying', 'quiet']
    T = pomdp_py.TabularTransitionModel({
        # state, action, next state
        ('hungry', 'feed',   'sated'):  1.0,
        ('hungry', 'feed',   'hungry'): 0.0,

        ('hungry', 'sing',   'hungry'): 1.0,
        ('hungry', 'sing',   'sated'):  0.0,

        ('hungry', 'ignore', 'hungry'): 1.0,
        ('hungry', 'ignore', 'sated'):  0.0,

        ('sated',  'feed',   'sated'):  1.0,
        ('sated',  'feed',   'hungry'): 0.0,

        ('sated',  'sing',   'hungry'): 0.1,
        ('sated',  'sing',   'sated'):  0.9,

        ('sated',  'ignore', 'hungry'): 0.1,
        ('sated',  'ignore', 'sated'):  0.9
    })

    O = pomdp_py.TabularObservationModel({
        # state, action, observation
        ('hungry', 'feed', 'crying'):  0.8,
        ('hungry', 'feed', 'quiet'):   0.2,

        ('hungry', 'sing', 'crying'):  0.9,
        ('hungry', 'sing', 'quiet'):   0.1,

        ('hungry', 'ignore', 'crying'): 0.8,
        ('hungry', 'ignore', 'quiet'):  0.2,

        ('sated', 'feed', 'crying'):   0.1,
        ('sated', 'feed', 'quiet'):    0.9,

        ('sated', 'sing', 'crying'):   0.1,
        ('sated', 'sing', 'quiet'):    0.9,

        ('sated', 'ignore', 'crying'): 0.1,
        ('sated', 'ignore', 'quiet'):  0.9,
    })

    R = pomdp_py.TabularRewardModel({
        # state, action
        ('hungry', 'feed'): -10 - 5,
        ('hungry', 'sing'): -10 - 0.5,
        ('hungry', 'ignore'): -10,

        ('sated', 'feed'): -5,
        ('sated', 'sing'): -0.5,
        ('sated', 'ignore'): 0
    })

    return S, A, Z, T, O, R

def cryingbaby_agent(hungry=0.22):
    S, A, Z, T, O, R = cryingbaby()
    pi = pomdp_py.UniformPolicyModel(A)
    b0 = pomdp_py.Histogram({"hungry": hungry,
                             "sated": 1.0 - hungry})
    agent = pomdp_py.Agent(b0, pi, T, O, R)
    return agent

def solve_with_vi_pruning(agent, discount_factor=0.9):
    horizon = 5

    filename = "cryingbaby.POMDP"
    pomdp_py.to_pomdp_file(agent, filename, discount_factor=discount_factor)

    # path to the pomdp-solve binary
    pomdp_solve_path = os.environ["POMDP_SOLVE_PATH"]
    if pomdp_solve_path is None or not os.path.exists(pomdp_solve_path):
        raise FileNotFoundError("To run this test, download pomdp-solve from "
                                "https://www.pomdp.org/code/. Then, follow the "
                                "instructions on this web page to compile this software. "
                                "Finally, set the environment variable POMDP_SOLVE_PATH "
                                "to be the path to the pomdp-solve binary file "
                                "generated after compilation, likely located at "
                                "/path/to/pomdp-solve-<version>/src/pomdp-solve ")
    policy = pomdp_py.vi_pruning(agent, pomdp_solve_path,
                                 discount_factor=discount_factor,
                                 options=["-horizon", horizon],
                                 remove_generated_files=False,
                                 return_policy_graph=False)
    return policy

def solve_with_sarsop(agent, discount_factor=0.9):
    pomdpsol_path = os.getenv("POMDPSOL_PATH")
    if pomdpsol_path is None or not os.path.exists(pomdpsol_path):
        raise FileNotFoundError("To run this test, download sarsop from"
                                "https://github.com/AdaCompNUS/sarsop. Then, follow the "
                                "instructions on this web page to compile this software. "
                                "Finally, set the environment variable POMDPSOL_PATH "
                                "to be the path to the pomdpsol binary file "
                                "generated after compilation, likely located at "
                                "/path/to/sarsop/src/pomdpsol")
    policy = pomdp_py.sarsop(agent, pomdpsol_path, discount_factor=discount_factor,
                             timeout=10, memory=20, precision=0.000001,
                             remove_generated_files=True,
                             logfile="solve_with_sarsop.log")
    return policy


def solve_with_pouct(agent, discount_factor=0.9):
    """POUCT is an online planner, so it's 'solved' during simulation-time"""
    return pomdp_py.POUCT(max_depth=5, discount_factor=discount_factor,
                          num_sims=4096, exploration_const=10,
                          rollout_policy=agent.policy_model,
                          show_progress=True)


def simulate_policy(agent, policy):
    # print(pomdp_py.value(agent.belief, S, A, Z, T, O, R, gamma, horizon=horizon))
    T = agent.transition_model
    R = agent.reward_model
    O = agent.observation_model

    state = "hungry"  # true initial state
    for step in range(10):
        action = policy.plan(agent)
        next_state = T.sample(state, action)
        reward = R.sample(state, action, next_state)
        observation = O.sample(next_state, action)

        # print
        print(f"step = {step+1}"
              f"\t|\taction: {action}"
              f"\t|\tobservation: {observation}"
              f"\t|\tstate: {state}  "
              f"\t|\treward: {reward}"
              f"\t|\tbelief: {agent.belief}")

        # update agent belief
        next_belief = pomdp_py.belief_update(agent.belief, action, observation, T, O)
        agent.set_belief(pomdp_py.Histogram(next_belief))

        # apply state transition to the environment
        state = next_state

def main():
    print(":::::::::::::::: VI PRUNING ::::::::::::::::")
    agent = cryingbaby_agent()
    policy = solve_with_vi_pruning(agent)
    simulate_policy(agent, policy)
    print(":::::::::::::::: SARSOP ::::::::::::::::")
    agent = cryingbaby_agent()
    policy = solve_with_sarsop(agent)
    simulate_policy(agent, policy)
    print(":::::::::::::::: POUCT ::::::::::::::::")
    agent = cryingbaby_agent()
    planner = solve_with_pouct(agent, discount_factor=0.9)
    simulate_policy(agent, planner)

if __name__ == "__main__":
    main()
