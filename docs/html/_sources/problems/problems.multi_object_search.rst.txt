problems.multi\_object\_search package
======================================

Multi-Object Search (MOS) Task
******************************

This task is based on the Multi-Object Search (MOS) task described in the
following paper

`Multi-Object Search using Object-Oriented POMDPs <https://h2r.cs.brown.edu/wp-content/uploads/wandzel19.pdf>`_ (ICRA 2019)

In this implementation, we consider a different (simpler) motion action scheme,
instead of based on topological graph and room connectivity.

.. autosummary::
   :nosignatures:

   ~problems.multi_object_search.problem
   ~problems.multi_object_search.example_worlds
   ~problems.multi_object_search.env.env
   ~problems.multi_object_search.domain
   ~problems.multi_object_search.domain.state
   ~problems.multi_object_search.domain.action
   ~problems.multi_object_search.domain.observation
   ~problems.multi_object_search.models
   ~problems.multi_object_search.models.transition_model   
   ~problems.multi_object_search.models.observation_model
   ~problems.multi_object_search.models.reward_model
   ~problems.multi_object_search.models.policy_model
   ~problems.multi_object_search.models.components.sensor
   ~problems.multi_object_search.models.components.grid_map   

   
problems.multi\_object\_search.problem module
---------------------------------------------

.. automodule:: problems.multi_object_search.problem
   :members:
   :undoc-members:
   :show-inheritance:

problems.multi\_object\_search.example\_worlds module
-----------------------------------------------------

.. automodule:: problems.multi_object_search.example_worlds
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::

   problems.multi_object_search.env
   problems.multi_object_search.domain   
   problems.multi_object_search.models

