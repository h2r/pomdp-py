"""Defines the PolicyModel for the continuous light-dark domain;
This is a continuous action space where actions are velocities.
Different from other discrete domains, there is no enumeration over
the action space provided, unless discretization is enforced.

The policy model was not explicit in the original paper `Belief space planning
assuming maximum likelihood observations` which is not proposing a sample
based algorithm. This is an example of how PolicyModel can be flexibly
used for non-MCTS or sampling based planners. Also, this PolicyModel indeed
supports a variant of POMCP that handles continuous domains, which we will
implement.
"""
import pomdp_py
import random
import pomdp_problems.light_dark as ld

class UniformPolicyModel(pomdp_py.RolloutPolicy):
    """In this policy model, actions are randomly
    chosen uniformly within a specified range."""
    def __init__(self, vx_range, vy_range):
        """
        vx_range (tuple): a tuple of two floats (vmin, vmax) to indicate the 
            range of velocity along the x direction
        vy_range (tuple): a tuple of two floats (vmin, vmax) to indicate the 
            range of velocity along the y direction
        """
        self._vx_range = vx_range
        self._vy_range = vy_range

    def sample(self, state, **kwargs):
        return (random.uniform(*self._vx_range),
                random.uniform(*self._vy_range))

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        raise NotImplementedError

    def rollout(self, state, history=None):
        # TODO: ignore history?
        return random.sample(state)
    
    
class DiscretePolicyModel(pomdp_py.PolicyModel):
    """This is a simple policy model, where there is a set of
    candidate velocities in each direction, making the action
    space discrete and finite."""
    def __init__(self, vx_vals, vy_vals):
        """
        vx_vals (sequence): a sequence of floats representing possible x velocities
        vx_vals (sequence): a sequence of floats representing possible y velocities
        """
        self._vx_vals = vx_vals
        self._vy_vals = vy_vals
        self._all_controls = set({})
        for vx in self._vx_vals:
            for vy in self._vx_vals:
                self._all_controls.add((vx,vy))

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        return set({ld.Action(control)
                    for control in self._all_controls})

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(history=history), 1)[0]
