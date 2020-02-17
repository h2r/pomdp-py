pomdp_problems.multi\_object\_search package
============================================

Multi-Object Search (MOS) Task
******************************

This task is based on the Multi-Object Search (MOS) task described in the
following paper

`Multi-Object Search using Object-Oriented POMDPs <https://h2r.cs.brown.edu/wp-content/uploads/wandzel19.pdf>`_ (ICRA 2019)

In this implementation, we consider a different (simpler) motion action scheme,
instead of based on topological graph and room connectivity.

.. autosummary::
   :nosignatures:

   ~pomdp_problems.multi_object_search.problem
   ~pomdp_problems.multi_object_search.example_worlds
   ~pomdp_problems.multi_object_search.env.env
   ~pomdp_problems.multi_object_search.domain
   ~pomdp_problems.multi_object_search.domain.state
   ~pomdp_problems.multi_object_search.domain.action
   ~pomdp_problems.multi_object_search.domain.observation
   ~pomdp_problems.multi_object_search.models
   ~pomdp_problems.multi_object_search.models.transition_model   
   ~pomdp_problems.multi_object_search.models.observation_model
   ~pomdp_problems.multi_object_search.models.reward_model
   ~pomdp_problems.multi_object_search.models.policy_model
   ~pomdp_problems.multi_object_search.models.components.sensor
   ~pomdp_problems.multi_object_search.models.components.grid_map   

   
pomdp_problems.multi\_object\_search.problem module
---------------------------------------------------

.. automodule:: pomdp_problems.multi_object_search.problem
   :members:
   :undoc-members:
   :show-inheritance:

pomdp_problems.multi\_object\_search.example\_worlds module
-----------------------------------------------------------

.. automodule:: pomdp_problems.multi_object_search.example_worlds
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::

   pomdp_problems.multi_object_search.env
   pomdp_problems.multi_object_search.domain   
   pomdp_problems.multi_object_search.models

