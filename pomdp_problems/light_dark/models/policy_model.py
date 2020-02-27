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
