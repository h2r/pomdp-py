"""
TODO: write Documentation
"""

import pomdp_py
import random
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


EPSILON = 1e-2

class State(pomdp_py.State):
    LOAD_LOCATION = 10
    def __init__(self, x, loaded):
        if type(x) != int or x < 0:
            raise ValueError("Invalid state: {}\n".format((x, loaded)) +
                             "x must be an integer > 0")
        if type(loaded) != bool:
            raise ValueError("Invalid state: {}\n".format((x, loaded)) +
                             "loaded must be a boolean")
        if x == 0 and loaded == True:
            raise ValueError("Agent can not be loaded in the 0th position")
        if x == self.LOAD_LOCATION and loaded == False:
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
    def __init__(self, obs):
        if obs not in ["load", "unload", "middle"]:
            raise ValueError("Invalid observation: {}\n".format(name) +
                             "Observation must be an integer > 0")
        self.name = obs
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return str(self.name)
    def __repr__(self):
        return "Observation(%s)" % str(self.x)

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    """This problem is small enough for the probabilities to be directly given
    externally"""
    def probability(self, observation, next_state, action, normalized=False, **kwargs):
        if observation != self.sample(next_state, action):
            return EPSILON
        else:
            return 1 - EPSILON

    def sample(self, next_state, action, normalized=False, **kwargs):
        if next_state.x == 0:
            return Observation("unload")
        elif next_state.x == State.LOAD_LOCATION:
            return Observation("load")
        else:
            return Observation("middle")

    def argmax(self, next_state, action, normalized=False, **kwargs):
        """Returns the most likely observation"""
        return self.sample(next_state, action)


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    """This problem is small enough for the probabilities to be directly given
            externally"""
    def probability(self, next_state, state, action, normalized=False, **kwargs):
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1 - EPSILON

    def sample(self, state, action, normalized=False, **kwargs):
        if ((state.x == State.LOAD_LOCATION and action == "move-right") or 
           (state.x == 0 and action == "move-left")):
            # trying to make invalid move, stay in the same place
            return state

        if action == "move-right":
            # make sure we're always loaded in the far right cell
            if state.x == State.LOAD_LOCATION - 1:
                return State(state.x + 1, True)
            return State(state.x + 1, state.loaded)

        if action == "move-left":
            # make sure we're always unloaded in the first cell
            if state.x == 1:
                return State(state.x - 1, False)
            return State(state.x - 1, state.loaded)

    def argmax(self, state, action, normalized=False, **kwargs):
        """Returns the most likely next state"""
        return self.sample(state, action)

# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self.sample(state, action):
            return 1.0
        else:
            return 0.0

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # if we are unloaded things, give reward 100, otherwise give -1
        if action == "move-left" and state.loaded == True and state.x == 1:
            return 100
        else:
            return -1

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        return self.sample(state, action)

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    def __init__(self):
        self._all_actions = {Action("move-right"), Action("move-left")}

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError  # Never used
    
    def sample(self, state, normalized=False, **kwargs):
        return self.get_all_actions().random()
    
    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError
    
    def get_all_actions(self, **kwargs):
        return self._all_actions

        
class LoadUnloadProblem(pomdp_py.POMDP):

    def __init__(self, init_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(),
                               RewardModel())

        env = pomdp_py.Environment(init_state,
                                   TransitionModel(),
                                   RewardModel())

        super().__init__(agent, env, name="LoadUnloadProblem")

def generate_random_state():
    # Flip a coin to determine if we are loaded
    loaded = np.random.rand() > 0.5
    location = np.random.randint(0, State.LOAD_LOCATION + 1)
    if location == 0:
        loaded = False
    if location == State.LOAD_LOCATION:
        loaded = True
    return State(location, loaded)

def generate_init_belief(num_particles):
    particles = []
    for _ in range(num_particles):
        particles.append(generate_random_state())

    return pomdp_py.Particles(particles)

def test_planner(load_unload_problem, planner, nsteps=3, discount=0.95):
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0

    fig = plt.figure()
    plt.title("Load/Unload problem (Red = empty, Blue = full)")
    plt.xlabel("Position")

    ax = fig.add_subplot(111)
    ax.set_xlim(-1, State.LOAD_LOCATION+1)
    ax.set_ylim(0, 2)
    x, y = [], []
    scat, = ax.plot(x, y, marker="x", markersize=20, ls=" ", color="black")
    
    def update(t):
        nonlocal gamma, total_reward, total_discounted_reward
        print("==== Step %d ====" % (t+1))
        action = planner.plan(load_unload_problem.agent)
        
        true_state = copy.deepcopy(load_unload_problem.env.state)
        env_reward = load_unload_problem.env.state_transition(action, execute=True)
        
        real_observation = load_unload_problem.env.provide_observation(
                load_unload_problem.agent.observation_model, action)
        load_unload_problem.agent.update_history(action, real_observation)
        planner.update(load_unload_problem.agent, action, real_observation)
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)

        new_x, new_y = [true_state.x], [1]
        scat.set_data(new_x, new_y)
        scat.set_color("b" if true_state.loaded else "r")
        return scat,

    ani = FuncAnimation(fig, update, frames=nsteps, interval=500)
    plt.show()

def main():
    # This is the length of the coridor the agent is moving up and down.
    # Valid states will be (x, loaded) where 0 <= x < state_count represents
    # the horizontal location of the agent, and loaded is a boolean.
    init_state = generate_random_state()
    init_belief = generate_init_belief(num_particles=200)
    load_unload_problem = LoadUnloadProblem(init_state, init_belief)

    print("** Testing POMCP **")
    pomcp = pomdp_py.POMCP(max_depth=100, discount_factor=0.95,
                           num_sims=100, exploration_const=110,
                           rollout_policy=load_unload_problem.agent.policy_model)
    test_planner(load_unload_problem, pomcp, nsteps=100)

if __name__ == '__main__':
    main()
