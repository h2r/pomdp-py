Using External Solvers
======================

.. automodule:: pomdp_py.utils.interfaces.solvers


|

.. contents:: **Table of Contents**
   :local:
   :depth: 1




Converting a pomdp_py :py:mod:`~pomdp_py.framework.basics.Agent` to a POMDP File
-------------------------------------------
Many existing libraries take as input a POMDP model written in a text file.
There are two file formats: :code:`.POMDP` `(link) <http://www.pomdp.org/code/pomdp-file-spec.html>`_ and :code:`.POMDPX` `(link) <https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation>`_. A :code:`.POMDP` file can be converted
into a :code:`.POMDPX` file using the :code:`pomdpconvert` program that is part of the `SARSOP toolkit <https://github.com/AdaCompNUS/sarsop>`_.

If a pomdp_py :py:mod:`~pomdp_py.framework.basics.Agent` has enumerable state :math:`S`, action :math:`A`, and observation spaces :math:`\Omega`, with explicitly defined probability for its models (:math:`T,O,R`), then it can be converted to either the POMDP file Format (:py:mod:`~pomdp_py.utils.interfaces.conversion.to_pomdp_file`) or the POMDPX file format (:py:mod:`~pomdp_py.utils.interfaces.conversion.to_pomdpx_file`).

.. autofunction:: pomdp_py.utils.interfaces.conversion.to_pomdp_file

.. autofunction:: pomdp_py.utils.interfaces.conversion.to_pomdpx_file

Example
~~~~~~~

Let's use the existing :py:mod:`~pomdp_problems.tiger.tiger_problem.Tiger` problem as an example.
First we create an instance of the Tiger problem:

.. code-block:: python

   from pomdp_problems.tiger.tiger_problem import TigerProblem, TigerState
   tiger = TigerProblem(0.15, TigerState(init_state),
                pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                    TigerState("tiger-right"): 0.5}))

convert to .POMDP file

.. code-block:: python

   from pomdp_py import to_pomdp_file
   filename = "./test_tiger.POMDP"
   to_pomdp_file(tiger.agent, filename, discount_factor=0.95)

convert to .POMDPX file

.. code-block:: python

   from pomdp_py import to_pomdpx_file
   filename = "./test_tiger.POMDPX"
   pomdpconvert_path = "~/software/sarsop/src/pomdpconvert"
   to_pomdpx_file(tiger.agent, pomdpconvert_path,
                  output_path=filename,
                  discount_factor=0.95)



Using pomdp-solve
-----------------
Pass in the agent to the :py:mod:`~pomdp_py.utils.interfaces.solvers.vi_pruning` function,
and it will run the `pomdp-solve` binary (using specified path)

.. autofunction:: pomdp_py.utils.interfaces.solvers.vi_pruning


Example
~~~~~~~

Computing a policy. We recommend using the :py:mod:`~pomdp_py.utils.interfaces.conversion.AlphaVectorPolicy`.

.. code-block:: python

   from pomdp_py import vi_pruning
   policy = vi_pruning(tiger.agent, pomdp_solve_path, discount_factor=0.95,
                       options=["-horizon", "100"],
                       remove_generated_files=False,
                       return_policy_graph=False)

Using the policy

.. code-block:: python

   for step in range(10):
        action = policy.plan(tiger.agent)
        reward = tiger.env.state_transition(action, execute=True)
        observation = tiger.agent.observation_model.sample(tiger.env.state, action)

        if isinstance(policy, PolicyGraph):
            policy.update(tiger.agent, action, observation)
        else:
            # AlphaVectorPOlicy
            # ... perform belief update on agent

Using sarsop
------------

.. autofunction:: pomdp_py.utils.interfaces.solvers.sarsop

Example
~~~~~~~

Computing a policy

.. code-block:: python

   from pomdp_py import sarsop
   policy = sarsop(tiger.agent, pomdpsol_path, discount_factor=0.95,
                   timeout=10, memory=20, precision=0.000001,
                   remove_generated_files=True)

Using the policy (Same as above, for the :py:mod:`pomdp_py.utils.interfaces.conversion.AlphaVectorPolicy` case)

.. code-block:: python

   for step in range(10):
        action = policy.plan(tiger.agent)
        reward = tiger.env.state_transition(action, execute=True)
        observation = tiger.agent.observation_model.sample(tiger.env.state, action)
        # ... perform belief update on agent


PolicyGraph and AlphaVectorPolicy
---------------------------------

PolicyGraph and AlphaVectorPolicy extend the :py:mod:`~pomdp_py.framework.planner.Planner`
interface which means they have a :code:`plan` function that can be used to
output an action given an agent (using the agent's belief).

.. autoclass:: pomdp_py.utils.interfaces.conversion.PolicyGraph
   :members: construct, plan, update


.. autoclass:: pomdp_py.utils.interfaces.conversion.AlphaVectorPolicy
   :members: construct, value, plan
