:orphan:

Design Principles
*****************


1. Distributions are the fundamental building blocks of a POMDP.

   Essentially, a POMDP describes the interaction between an `agent` and the
   `environment`.  The interaction is formally encapsulated by a few important
   `generative probability distributions`. The core of pomdp_py is built around
   this understanding. The interfaces in :py:mod:`pomdp_py.framework.basics`
   convey this idea. Using distributions as the building block avoids the 
   requirement of explicitly enumerating over :math:`S,A,O`. 

--

2. POMDP = agent + environment

   Like above, the gist of a POMDP is captured by the generative probability
   distributions including the
   :class:`~pomdp_py.framework.basics.TransitionModel`,
   :class:`~pomdp_py.framework.basics.ObservationModel`,
   :class:`~pomdp_py.framework.basics.RewardModel`. Because, generally, :math:`T, R, O`
   may be different for the agent versus the environment (to support learning),
   it does not make much sense to have the POMDP class to hold this information;
   instead, Agent should have its own :math:`T, R, O, \pi` and the Environment should
   have its own :math:`T, R`. The job of a POMDP is only to verify whether a given
   state, action, or observation are valid. See :class:`~pomdp_py.framework.basics.Agent`
   and :class:`~pomdp_py.framework.basics.Environment`.


.. figure:: pomdp.jpg
   :alt: POMDP diagram

   A Diagram for POMDP :math:`\langle S,A,\Omega,T,O,R \rangle`. (**correction**:
   change :math:`o\in S` to :math:`o\in\Omega`)

