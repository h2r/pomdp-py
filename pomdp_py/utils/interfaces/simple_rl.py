"""
Provides utility functions that interfaces with `simple_rl <https://github.com/david-abel/simple_rl>`_.

Essentially, this will convert an agent in pomdp_py into a simple_rl.MDPClass
or POMDPClass. Note that since creating these classes require enumerable
aciton and observation spaces, this conversion is only feasible for agents
whose ObservationModel and PolicyModel can return a list of all observations /
actions.

Note: simple_rl is a library for Reinforcement Learning developed and
maintained by David Abel. It is also an early-stage library.

Warning:
simple_rl is simple_rl's POMDP functionality is currently relatively
lacking. Providing this inteface is mostly to potentially leverage the MDP
algorithms in simple_rl.
"""
import simple_rl

def convert_to_MDPClass(pomdp, discount_factor=0.99, step_cost=0):
    """Converts the pomdp to the building block MDPClass of simple_rl.  There are a lot of
    variants of MDPClass in simple_rl. It is up to the user to then convert this
    MDPClass into those variants, if necessary.

    Clearly, if this agent is partially observable, this conversion
    will change the problem and make it no longer a POMDP."""
    agent = pomdp.agent
    env = pomdp.env
    try:
        all_actions = agent.policy_model.get_all_actions()
    except NotImplementedError:
        raise ValueError("This agent does not have enumerable action space.")

    # Since we do not know how env.state is represented, we
    # cannot turn it into a simple_rl State with "features",
    # since the features must be represented as a list; In
    # any case, the user, with knowledge of the state
    # representaion, could later convert it into the format
    # that simple_rl is supposed to work with.
    state = simple_rl.State(data=env.state)

    return simple_rl.MDP(all_actions,
                         agent.transition_model.sample,
                         agent.reward_model.sample,
                         gamma=discount_factor,
                         step_cost=step_cost)


def convert_to_POMDPClass(pomdp,
                          discount_factor=0.99, step_cost=0,
                          belief_updater_type="discrete"):
    agent = pomdp.agent
    env = pomdp.env
    try:
        all_actions = agent.policy_model.get_all_actions()
    except NotImplementedError:
        raise ValueError("This agent does not have enumerable action space.")
    try:
        all_observations = agent.observation_model.get_all_observations()
    except NotImplementedError:
        raise ValueError("This agent does not have enumerable observation space.")

    try:
        belief_hist = agent.belief.get_histogram()
    except Exception:
        raise ValueError("Agent belief cannot be converted into a histogram;\n"
                         "thus cannot create POMDPClass.")

    return simple_rl.POMDP(all_actions,
                           all_observations,
                           agent.transition_model.sample,
                           agent.reward_model.sample,
                           agent.observation_model.sample,
                           belief_hist,
                           belief_updater_type=belief_updater_type,
                           gamma=discount_factor,
                           step_cost=step_cost)
