"""
PO-rollout: Baseline algorithm in the POMCP paper :cite:`silver2010monte`.

Quote from the POMCP paper:

    To provide a performance benchmark in these cases, we evaluated the
    performance of simple Monte-Carlo simulation without any tree.
    The PO-rollout algorithm used Monte-Carlo belief state updates,
    as described in section 3.2. It then simulated :math:`n/|A|` rollouts for
    each legal action, and selected the action with highest average return.

We don't require Monte-Carlo belief update (it's an option). But
it will do the rollouts and action selection as described.
"""

from pomdp_py.framework.basics cimport Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel,\
    sample_generative_model, Response
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.representations.belief.particles cimport particle_reinvigoration
from pomdp_py.algorithms.po_uct cimport RandomRollout
# from pomdp_py.algorithms.pomcp cimport VNodeParticles, RootVNodeParticles
import copy
import time
import random
import math

cdef class PORollout(Planner):

    """
    PO-rollout: Baseline algorithm in the POMCP paper
    """

    def __init__(self,
                 num_sims=100,
                 max_depth=5, discount_factor=0.9,
                 rollout_policy=RandomRollout(),
                 particles=False,  # true if use Monte-Carlo belief update
                 action_prior=None):
        self._num_sims = num_sims
        self._max_depth = max_depth
        self._rollout_policy = rollout_policy
        self._action_prior = action_prior
        self._discount_factor = discount_factor
        self._particles = particles

        self._agent = None
        self._last_best_response = None

    @property
    def last_best_response(self):
        return self._last_best_response

    cpdef public plan(self, Agent agent):
        self._agent = agent
        best_action, best_response = self._search()
        self._last_best_response = best_response
        return best_action

    cpdef _search(self):
        cdef Action best_action
        cdef Response best_response
        cdef Response response_avg
        cdef Response total_discounted_response
        cdef set legal_actions
        cdef list responses

        best_action, best_response = None, Response(float("-inf"))
        legal_actions = self._agent.valid_actions(history=self._agent.history)
        for action in legal_actions:
            responses = []
            for i in range(self._num_sims // len(legal_actions)):
                state = self._agent.belief.random()
                total_discounted_response = self._rollout(state, 0)
                responses.append(total_discounted_response)
            response_avg = sum(responses) / len(responses)
            if response_avg > best_response:
                best_action = action
                best_response = response_avg
        return best_action, best_response

    cpdef _rollout(self, State state, int depth):
        # Rollout without a tree.
        cdef Action action
        cdef float discount = 1.0
        cdef Response total_discounted_response = Response()
        cdef State next_state
        cdef Observation observation
        cdef Response response
        cdef int nsteps
        cdef tuple history = self._agent.history

        while depth <= self._max_depth:
            action = self._rollout_policy.rollout(state, history=history)
            next_state, observation, response, nsteps = sample_generative_model(self._agent, state, action)
            history = history + ((action, observation),)
            depth += 1
            total_discounted_response = total_discounted_response + response * discount
            discount *= self._discount_factor
            state = next_state
        return total_discounted_response

    cpdef update(self, Agent agent, Action real_action, Observation real_observation,
                 state_transform_func=None):
        # If particles is true, then perform Monte Carlo belief update.
        # Otherwise, do nothing
        cdef int nsteps
        if self._particles:
            cur_belief = agent.belief
            new_belief = Particles([])
            if not isinstance(cur_belief, Particles):
                raise ValueError("Agent's belief is not in particles.")
            for state in cur_belief.particles:
                next_state, observation, response, nsteps = sample_generative_model(agent, state,
                                                                                  real_action)
                if observation == real_observation:
                    new_belief.add(next_state)
            # Particle reinvigoration
            agent.set_belief(particle_reinvigoration(new_belief,
                                                     len(agent.init_belief.particles),
                                                     state_transform_func=state_transform_func))

    @property
    def update_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return self._particles

    def clear_agent(self):
        """clear_agent(self)"""
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_best_response = Response(float('-inf'))
        
    cpdef set_rollout_policy(self, RolloutPolicy rollout_policy):
        """
        set_rollout_policy(self, RolloutPolicy rollout_policy)
        Updates the rollout policy to the given one
        """
        self._rollout_policy = rollout_policy
