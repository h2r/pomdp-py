# This is a POMDP problem; Namely, it specifies both
# the POMDP (i.e. state, action, observation space)
# and the T/O/R for the agent as well as the environment.

# The description of the tiger problem is as follows:
# (Quote from __POMDP: Introduction to Partially Observable Markov
#  Decision Processes__ by Kamalzadeh and Hahsler
#  https://cran.r-project.org/web/packages/pomdp/vignettes/POMDP.pdf)

# A tiger is put with equal probability behind one
# of two doors, while treasure is put behind the other one.
# You are standing in front of the two closed doors and
# need to decide which one to open. If you open the door
# with the tiger, you will get hurt (negative reward).
# But if you open the door with treasure, you receive
# a positive reward. Instead of opening a door right away,
# you also have the option to wait and listen for tiger noises. But
# listening is neither free nor entirely accurate. You might hear the
# tiger behind the left door while it is actually behind the right
# door and vice versa.

# States: tiger-left, tiger-right
# Actions: open-left, open-right, listen
# Rewards: +10 for opening treasure door. -100 for opening tiger door.
#          -1 for listening.
# Observations: You can hear either "tiger-left", or "tiger-right".

import pomdp_py
import random
import numpy as np
import sys

class TigerProblem:

    STATES = {"tiger-left", "tiger-right"}
    ACTIONS = {"open-left", "open-right", "listen"}
    OBSERVATIONS = {"open-left", "open-right", "listen"}

    def __init__(self, obs_probs, trans_probs, init_true_state, init_belief):
        """init_belief is a Distribution."""
        self._obs_probs = obs_probs
        self._trans_probs = trans_probs
        assert TigerProblem.POMDP.verify_state(init_true_state)
        agent = pomdp_py.Agent(TigerProblem.POMDP, init_belief,
                               TigerProblem.PolicyModel(),
                               TigerProblem.TransitionModel(self._trans_probs),
                               TigerProblem.ObservationModel(self._obs_probs),
                               TigerProblem.RewardModel())
        env = pomdp_py.Environment(TigerProblem.POMDP,
                                   init_true_state,
                                   TigerProblem.TransitionModel(self._trans_probs),
                                   TigerProblem.RewardModel())
        self._agent = agent
        self._env = env

    class POMDP(pomdp_py.pomdp):
        @classmethod
        def verify_state(cls, state):
            return state in TigerProblem.STATES

        @classmethod
        def verify_action(cls, action):
            return action in TigerProblem.ACTIONS

        @classmethod
        def verify_observation(cls, observation):
            return observation in TigerProblem.OBSERVATIONS
        
    # Observation model
    class ObservationModel(pomdp_py.ObservationModel):
        """This problem is small enough for the probabilities to be directly given
        externally"""
        def __init__(self, probs):
            self._probs = probs

        def probability(self, observation, next_state, action, normalized=False, **kwargs):
            return self._probs[next_state][action][observation]

        def sample(self, next_state, action, normalized=False, **kwargs):
            """Returns a tuple, (observation, probability) """
            return self.get_distribution(next_state, action).random()

        def argmax(self, next_state, action, normalized=False, **kwargs):
            """Returns the most likely observation"""
            return max(self._probs[next_state][action], key=self._probs[next_state][action].get)

        def get_distribution(self, next_state, action, **kwargs):
            """Returns the underlying distribution of the model; In this case, it's just a histogram"""
            return pomdp_py.Histogram(self._probs[next_state][action])
        
    # Transition Model
    class TransitionModel(pomdp_py.TransitionModel):
        """This problem is small enough for the probabilities to be directly given
                externally"""
        def __init__(self, probs):
            self._probs = probs

        def probability(self, next_state, state, action, normalized=False, **kwargs):
            return self._probs[state][action][next_state]

        def sample(self, state, action, normalized=False, **kwargs):
            """Returns a tuple, (next_staet, probability) """
            return self.get_distribution(state, action).random()
            
        def argmax(self, state, action, normalized=False, **kwargs):
            """Returns the most likely next state"""
            return max(self._probs[state][action], key=self._probs[state][action].get) 

        def get_distribution(self, state, action, **kwargs):
            """Returns the underlying distribution of the model"""
            return pomdp_py.Histogram(self._probs[state][action])

    # Reward Model
    class RewardModel(pomdp_py.RewardModel):
        def __init__(self, scale=1):
            self._scale = scale
        def _reward_func(self, state, action, next_state):
            reward = 0
            if action == "open-left":
                if state== "tiger-right":
                    reward += 10 * self._scale
                else:
                    reward -= 100 * self._scale
            elif action == "open-right":
                if state== "tiger-left":
                    reward += 10 * self._scale
                else:
                    reward -= 100 * self._scale
            elif action == "listen":
                reward -= 1 * self._scale
            return reward

        def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
            if reward == self._reward_func(state, action, next_state):
                return 1.0
            else:
                return 0.0
        @abstractmethod    
        def sample(self, state, action, next_state, normalized=False, **kwargs):
            """Returns a reward"""
            # deterministic
            return self._reward_func(state, action, next_state)
        @abstractmethod
        def argmax(self, state, action, next_state, normalized=False, **kwargs):
            """Returns the most likely reward"""
            return self._reward_func(state, action, next_state)
        @abstractmethod    
        def get_distribution(self, state, action, next_state, **kwargs):
            """Returns the underlying distribution of the model"""
            reward = self._reward_func(state, action, next_state)
            return pomdp_py.Histogram({reward:1.0})

    # Policy Model
    class PolicyModel(pomdp_py.PolicyModel):
        """This is an extremely dumb policy model; To keep consistent
        with the framework."""
        def __init__(self, prior=None):
            self._prior = {}
            self._probs = {}
            if prior is not None:
                self._prior = prior
                
        def probability(self, action, state, normalized=False, **kwargs):
            if state not in self._probs:
                if action in self._prior:
                    return self._prior[action]
                else:
                    return 1.0/len(TigerProblem.ACTIONS)
            else:
                return self._probs[state][action]
        
        def sample(self, state, normalized=False, **kwargs):
            return self.get_distribution(state).random()

        def argmax(self, state, normalized=False, **kwargs):
            """Returns the most likely reward"""
            raise NotImplemented

        def get_distribution(self, state, **kwargs):
            """Returns the underlying distribution of the model"""
            return pomdp_py.Histogram(self._probs[state])
