Use Case Extensions
===================

The interfaces provided by pomdp_py should support projects in these directions (and more):

1. POMDP for RL (Reinforcement Learning)

   - Learn a :class:`~pomdp_py.framework.basics.PolicyModel` (model-free)

   - Learn a :class:`~pomdp_py.framework.basics.TransitionModel` and
     :class:`~pomdp_py.framework.basics.ObservationModel` (model-based)

2. Multi-Agent POMDPs

   - Define multiple :class:`~pomdp_py.framework.basics.Agent` classes.

3. Object-Oriented POMDPs

   - The Object-Oriented POMDP (OO-POMDP) already has its interfaces implemented
     in :py:mod:`~pomdp_py.framework.oopomdp`.

3. Task Transfer or Transfer Learning

   - Define multiple :class:`~pomdp_py.framework.basics.Agent` classes.

3. Planning Algorithm Research

   - Use existing POMCP or POUCT as baslines.
