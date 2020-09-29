"""
TODO: write Documentation
"""

import pomdp_py
import random
import numpy as np
import sys

def build_states(tuples):
    return {State(*t) for t in tuples}
def build_actions(strings):
    return {Action(s) for s in strings}
def build_observations(ints):
    return {Observation(x) for x in ints}

class State(pomdp_py.State):
    state_count = 10
    def __init__(self, x, loaded):
        if type(x) != int or x < 0:
            raise ValueError("Invalid state: {}\n".format((x, loaded)) +
                             "x must be an integer > 0")
        if type(loaded) != bool:
            raise ValueError("Invalid state: {}\n".format((x, loaded)) +
                             "loaded must be a boolean")
        if x == 0 and loaded == True:
            raise ValueError("Agent can not be loaded in the 0th position")
        if x == self.state_count and loaded == False:
            raise ValueError("Agent can not be unloaded in the last position")

        self.x = x
        self.loaded = loaded
    def __hash__(self):
        return hash((self.x, self.loaded))
    def __eq__(self, other):
        if isinstance(other, State):
            return self.x == other.x and self.loaded == self.loaded
        elif type(other) == tuple:
            return self.x == other[0] and self.loaded == other[1]
    def __str__(self):
        return str((self.x, self.loaded))
    def __repr__(self):
        return "State({})".format(self)
    
class Action(pomdp_py.Action):
    def __init__(self, name):
        if name != "move-left" and name != "move-right":
            raise ValueError("Invalid action: %s" % name)        
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name
    
class Observation(pomdp_py.Observation):
    def __init__(self, x):
        if type(x) != int:
            raise ValueError("Invalid observation: {}\n".format(name) +
                             "Observation must be an integer > 0")
        self.x = x
    def __hash__(self):
        return hash(self.x)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.x == other.x
        elif type(other) == int:
            return self.x == other
    def __str__(self):
        return str(self.x)
    def __repr__(self):
        return "Observation(%s)" % str(self.x)

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    """This problem is small enough for the probabilities to be directly given
    externally"""
    def __init__(self, probs):
        self._probs = probs

    def probability(self, observation, next_state, action, normalized=False, **kwargs):
        try:
            return self._probs[next_state][action][observation]
        except:
            return 0

    def sample(self, next_state, action, normalized=False, **kwargs):
        return Observation(next_state.x)

    def argmax(self, next_state, action, normalized=False, **kwargs):
        """Returns the most likely observation"""
        return max(self._probs[next_state][action], key=self._probs[next_state][action].get)

    def get_distribution(self, next_state, action, **kwargs):
        """Returns the underlying distribution of the model; In this case, it's just a histogram"""
        return pomdp_py.Histogram(self._probs[next_state][action])

    def get_all_observations(self):
        return LoadUnloadProblem.OBSERVATIONS

# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    """This problem is small enough for the probabilities to be directly given
            externally"""
    def __init__(self, probs):
        self._probs = probs

    def probability(self, next_state, state, action, normalized=False, **kwargs):
        try:
            return self._probs[state][action][next_state]
        except:
            return 0

    def sample(self, state, action, normalized=False, **kwargs):
        if ((state.x == State.state_count - 1 and action == "move-right") or 
           (state.x == 0 and action == "move-left")):
            # trying to make invalid move, stay in the same place
            return state

        if action == "move-right":
            # make sure we're always loaded in the far right cell
            if state.x == State.state_count - 2:
                return State(state.x + 1, True)
            return State(state.x + 1, state.loaded)
        if action == "move-left":
            # make sure we're always unloaded in the first cell
            if state.x == 1:
                return State(state.x - 1, False)
            return State(state.x - 1, state.loaded)

    def argmax(self, state, action, normalized=False, **kwargs):
        """Returns the most likely next state"""
        return max(self._probs[state][action], key=self._probs[state][action].get) 

    def get_distribution(self, state, action, **kwargs):
        """Returns the underlying distribution of the model"""
        return pomdp_py.Histogram(self._probs[state][action])

    def get_all_states(self):
        return LoadUnloadProblem.STATES

# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, scale=1):
        self._scale = scale
    def _reward_func(self, state, action):
        # if we are unloaded things, give reward 100, otherwise give -1
        if action == "move-left" and state.loaded == True and state.x == 1:
            return 100
        else:
            return -1

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        return self._reward_func(state, action)

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        return self._reward_func(state, action)

    def get_distribution(self, state, action, next_state, **kwargs):
        """Returns the underlying distribution of the model"""
        reward = self._reward_func(state, action)
        return pomdp_py.Histogram({reward:1.0})

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError  # Never used
    
    def sample(self, state, normalized=False, **kwargs):
        return self.get_all_actions().random()
    
    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError
    
    def get_all_actions(self, **kwargs):
        return LoadUnloadProblem.ACTIONS

        
class LoadUnloadProblem(pomdp_py.POMDP):

    STATES = build_states(
        list([(i, False) for i in range(0, State.state_count-1)]) +
        list([(i, True) for i in range(1, State.state_count)]),
    )
    ACTIONS = build_actions({"move-left", "move-right"})
    OBSERVATIONS = build_observations([i for i in range(0, State.state_count)])

    def __init__(self, obs_probs, trans_probs, init_true_state, init_belief):
        """init_belief is a Distribution."""
        self._obs_probs = obs_probs
        self._trans_probs = trans_probs

        
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(self._trans_probs),
                               ObservationModel(self._obs_probs),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(self._trans_probs),
                                   RewardModel())
        super().__init__(agent, env, name="LoadUnloadProblem")


def test_planner(load_unload_problem, planner, nsteps=3):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        load_unload_problem (LoadUnloadProblem): an instance of the load unload problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    for i in range(nsteps):
        action = planner.plan(load_unload_problem.agent)
        print("==== Step %d ====" % (i+1))
        print("True state: %s" % load_unload_problem.env.state)
        print("Belief: %s" % str(load_unload_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(load_unload_problem.env.reward_model.sample(load_unload_problem.env.state, action, None)))

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        load_unload_problem.env.state_transition(action)
        real_observation = Observation(load_unload_problem.env.state.x)
        print(">> Observation: %s" % real_observation)
        
        planner.update(load_unload_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)
        if isinstance(load_unload_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(load_unload_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          load_unload_problem.agent.observation_model,
                                                          load_unload_problem.agent.transition_model)
            load_unload_problem.agent.set_belief(new_belief)

def build_setting(state_count):
    """There are 2*state_count-2 valid states (an agent can be in any location
    both loaded or unloaded except for the first location where the agent can
    only be loaded and the last location where the agent can only be loaded.
    """
    obs_probs = {}
    trans_probs = {}

    # The agent only gets observations about where it is, not whether it is
    # loaded
    for i in range(state_count):
        # For observations there are only two options of action, observation
        # pairs, the agent can move to the left and the right and then observe
        # their new location
        if i > 0:
            obs_probs[State(i, True)] = {}
            trans_probs[State(i, True)] = {}
        if i < state_count - 1:
            obs_probs[State(i, False)] = {}
            trans_probs[State(i, False)] = {}

        if i == 0:
            # Moving to the left from 0 doesn't move you
            obs_probs[State(i, False)][Action("move-right")] = {Observation(i+1): 1.0}
            obs_probs[State(i, False)][Action("move-left")] = {Observation(i): 1.0}
        elif i == state_count-1:
            # Moving to the right from end doesn't move you
            obs_probs[State(i, True)][Action("move-left")] = {Observation(i-1): 1.0}
            obs_probs[State(i, True)][Action("move-right")] = {Observation(i): 1.0}
        else:
            obs_probs[State(i, True)][Action("move-right")] = {Observation(i+1): 1.0}
            obs_probs[State(i, True)][Action("move-left")] = {Observation(i-1): 1.0}
            obs_probs[State(i, False)][Action("move-right")] = {Observation(i+1): 1.0}
            obs_probs[State(i, False)][Action("move-left")] = {Observation(i-1): 1.0}

        # For the trans_probs, the agent can move either left or right and the
        # loadedness stays the same unless they are entering the load or
        # unload zone
        if i == 0:
            # Moving to the left from 0 doesn't move you
            trans_probs[State(i, False)][Action("move-right")] = {State(i+1, False): 1.0}
            trans_probs[State(i, False)][Action("move-left")] = {State(i, False): 1.0}
        elif i == state_count-1:
            # Moving to the right from end doesn't move you
            trans_probs[State(i, True)][Action("move-left")] = {State(i-1, True): 1.0}
            trans_probs[State(i, True)][Action("move-right")] = {State(i, True): 1.0}
        elif i == 1:
            # Moving into the first cell unloads the agent
            trans_probs[State(i, True)][Action("move-left")] = {State(i-1, False): 1.0}
            trans_probs[State(i, True)][Action("move-right")] = {State(i+1, True): 1.0}
            trans_probs[State(i, False)][Action("move-left")] = {State(i-1, False): 1.0}
            trans_probs[State(i, False)][Action("move-right")] = {State(i+1, False): 1.0}
        elif i == state_count - 2:
            # Moving into the left cell loads the agent
            trans_probs[State(i, True)][Action("move-left")] = {State(i-1, True): 1.0}
            trans_probs[State(i, True)][Action("move-right")] = {State(i+1, True): 1.0}
            trans_probs[State(i, False)][Action("move-left")] = {State(i-1, False): 1.0}
            trans_probs[State(i, False)][Action("move-right")] = {State(i+1, True): 1.0}
        else:
            trans_probs[State(i, True)][Action("move-left")] = {State(i-1, True): 1.0}
            trans_probs[State(i, True)][Action("move-right")] = {State(i+1, True): 1.0}
            trans_probs[State(i, False)][Action("move-left")] = {State(i-1, False): 1.0}
            trans_probs[State(i, False)][Action("move-right")] = {State(i+1, False): 1.0}

    return {
        "obs_probs": obs_probs,
        "trans_probs": trans_probs
    }
            
def main():
    # This is the length of the coridor the agent is moving up and down.
    # Valid states will be (x, loaded) where 0 <= x < state_count represents
    # the horizontal location of the agent, and loaded is a boolean.
    setting = build_setting(State.state_count)

    init_true_state = State(0, False)
    init_belief = pomdp_py.Histogram({State(0, False): 1})
    load_unload_problem = LoadUnloadProblem(setting['obs_probs'],
                                 setting['trans_probs'],
                                 init_true_state, init_belief)

    print("** Testing POMCP **")
    load_unload_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=1), prior=True)
    pomcp = pomdp_py.POMCP(max_depth=50, discount_factor=0.95,
                           num_sims=100, exploration_const=110,
                           rollout_policy=load_unload_problem.agent.policy_model)
    test_planner(load_unload_problem, pomcp, nsteps=100)
    
    pomdp_py.visual.visualize_pouct_search_tree(load_unload_problem.agent.tree,
                                                max_depth=5, anonymize=False)

if __name__ == '__main__':
    main()
