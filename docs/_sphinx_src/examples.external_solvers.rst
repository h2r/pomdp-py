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
   init_state = "tiger-left"
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

**Setting the path.** After downloading and installing `pomdp-solve <https://www.pomdp.org/code/>`_,
a binary executable called :code:`pomdp-solve` should appear under :code:`path/to/pomdp-solve-<version>/src/`. We create a variable in Python
to point to this path:

.. code-block:: python

   pomdp_solve_path = "path/to/pomdp-solve-<version>/src/pomdp-solve"

**Computing a policy.** We recommend using the :py:mod:`~pomdp_py.utils.interfaces.conversion.AlphaVectorPolicy`;
That means setting :code:`return_policy_graph` to False (optional).

.. code-block:: python

   from pomdp_py import vi_pruning
   policy = vi_pruning(tiger.agent, pomdp_solve_path, discount_factor=0.95,
                       options=["-horizon", "100"],
                       remove_generated_files=False,
                       return_policy_graph=False)

**Using the policy.** Here the code checks whether the policy is a  :py:mod:`~pomdp_py.utils.interfaces.conversion.AlphaVectorPolicy` or a  :py:mod:`~pomdp_py.utils.interfaces.conversion.PolicyGraph`

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

|
| Complete example code:

.. code-block:: python

   import pomdp_py
   from pomdp_py import vi_pruning
   from pomdp_problems.tiger.tiger_problem import TigerProblem, TigerState

   # Initialize problem
   init_state = "tiger-left"
   tiger = TigerProblem(0.15, TigerState(init_state),
                pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                    TigerState("tiger-right"): 0.5}))

   # Compute policy
   pomdp_solve_path = "path/to/pomdp-solve-<version>/src/pomdp-solve"
   policy = vi_pruning(tiger.agent, pomdp_solve_path, discount_factor=0.95,
                       options=["-horizon", "100"],
                       remove_generated_files=False,
                       return_policy_graph=False)

   # Simulate the POMDP using the policy
   for step in range(10):
        action = policy.plan(tiger.agent)
        reward = tiger.env.state_transition(action, execute=True)
        observation = tiger.agent.observation_model.sample(tiger.env.state, action)
        print(tiger.agent.cur_belief, action, observation, reward)

        if isinstance(policy, pomdp_py.PolicyGraph):
            # No belief update needed. Just update the policy graph
            policy.update(tiger.agent, action, observation)
        else:
            # belief update is needed for AlphaVectorPolicy
            new_belief = pomdp_py.update_histogram_belief(tiger.agent.cur_belief,
                                                          action, observation,
                                                          tiger.agent.observation_model,
                                                          tiger.agent.transition_model)
            tiger.agent.set_belief(new_belief)


| Expected output (or similar):

.. code-block:: text

    //****************\\
   ||   pomdp-solve    ||
   ||     v. 5.4       ||
    \\****************//
         PID=8239
   - - - - - - - - - - - - - - - - - - - -
   time_limit = 0
   mcgs_prune_freq = 100
   verbose = context
   ...
   horizon = 100
   ...
   - - - - - - - - - - - - - - - - - - - -
   [Initializing POMDP ... done.]
   [Initial policy has 1 vectors.]
   ++++++++++++++++++++++++++++++++++++++++
   Epoch: 1...3 vectors in 0.00 secs. (0.00 total) (err=inf)
   Epoch: 2...5 vectors in 0.00 secs. (0.00 total) (err=inf)
   Epoch: 3...9 vectors in 0.00 secs. (0.00 total) (err=inf)
   ...
   Epoch: 95...9 vectors in 0.00 secs. (2.39 total) (err=inf)
   Epoch: 96...9 vectors in 0.00 secs. (2.39 total) (err=inf)
   Epoch: 97...9 vectors in 0.00 secs. (2.39 total) (err=inf)
   Epoch: 98...9 vectors in 0.00 secs. (2.39 total) (err=inf)
   Epoch: 99...9 vectors in 0.00 secs. (2.39 total) (err=inf)
   Epoch: 100...9 vectors in 0.01 secs. (2.40 total) (err=inf)
   ++++++++++++++++++++++++++++++++++++++++
   Solution found.  See file:
           temp-pomdp.alpha
           temp-pomdp.pg
   ++++++++++++++++++++++++++++++++++++++++
   User time = 0 hrs., 0 mins, 2.40 secs. (= 2.40 secs)
   System time = 0 hrs., 0 mins, 0.00 secs. (= 0.00 secs)
   Total execution time = 0 hrs., 0 mins, 2.40 secs. (= 2.40 secs)

   ** Warning **
           lp_solve reported 2 LPs with numerical instability.
   [(TigerState(tiger-right), 0.5), (TigerState(tiger-left), 0.5)] listen tiger-left -1.0
   [(TigerState(tiger-left), 0.85), (TigerState(tiger-right), 0.15)] listen tiger-left -1.0
   [(TigerState(tiger-left), 0.9697986575573173), (TigerState(tiger-right), 0.03020134244268276)] open-right tiger-left 10.0
   ...


..
  SARSOP

Using sarsop
------------

.. autofunction:: pomdp_py.utils.interfaces.solvers.sarsop

Example
~~~~~~~

**Setting the path.** After building `SARSOP <https://github.com/AdaCompNUS/sarsop>_`, there is a
binary file :code:`pomdpsol` under :code:`path/to/sarsop/src`.
We create a variable in Python
to point to this path:

.. code-block:: python

   pomdpsol_path = "path/to/sarsop/src/pomdpsol"

**Computing a policy.**

.. code-block:: python

   from pomdp_py import sarsop
   policy = sarsop(tiger.agent, pomdpsol_path, discount_factor=0.95,
                   timeout=10, memory=20, precision=0.000001,
                   remove_generated_files=True)

**Using the policy.** (Same as above, for the :py:mod:`~pomdp_py.utils.interfaces.conversion.AlphaVectorPolicy` case)

.. code-block:: python

   for step in range(10):
        action = policy.plan(tiger.agent)
        reward = tiger.env.state_transition(action, execute=True)
        observation = tiger.agent.observation_model.sample(tiger.env.state, action)
        # ... perform belief update on agent

|
| Complete example code:

.. code-block:: python

   import pomdp_py
   from pomdp_py import sarsop
   from pomdp_problems.tiger.tiger_problem import TigerProblem, TigerState

   # Initialize problem
   init_state = "tiger-left"
   tiger = TigerProblem(0.15, TigerState(init_state),
                pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                    TigerState("tiger-right"): 0.5}))

   # Compute policy
   pomdpsol_path = "path/to/sarsop/src/pomdpsol"
   policy = sarsop(tiger.agent, pomdpsol_path, discount_factor=0.95,
                   timeout=10, memory=20, precision=0.000001,
                   remove_generated_files=True)

   # Simulate the POMDP using the policy
   for step in range(10):
        action = policy.plan(tiger.agent)
        reward = tiger.env.state_transition(action, execute=True)
        observation = tiger.agent.observation_model.sample(tiger.env.state, action)
        print(tiger.agent.cur_belief, action, observation, reward)

        # belief update is needed for AlphaVectorPolicy
        new_belief = pomdp_py.update_histogram_belief(tiger.agent.cur_belief,
                                                      action, observation,
                                                      tiger.agent.observation_model,
                                                      tiger.agent.transition_model)
        tiger.agent.set_belief(new_belief)

.. code-block:: text

   Loading the model ...
     input file   : ./temp-pomdp.pomdp
     loading time : 0.00s

   SARSOP initializing ...
     initialization time : 0.00s

   -------------------------------------------------------------------------------
    Time   |#Trial |#Backup |LBound    |UBound    |Precision  |#Alphas |#Beliefs
   -------------------------------------------------------------------------------
    0       0       0        -20        92.8205    112.821     4        1
    0       2       51       -6.2981    63.7547    70.0528     7        15
    0       4       103      2.35722    52.3746    50.0174     5        19
    0       6       155      6.44093    45.1431    38.7021     5        20
    0       8       205      12.1184    36.4409    24.3225     5        20
    ...
    0       40      1255     19.3714    19.3714    7.13808e-06 5        21
    0       41      1300     19.3714    19.3714    3.76277e-06 6        21
    0       42      1350     19.3714    19.3714    1.75044e-06 12       21
    0       43      1393     19.3714    19.3714    9.22729e-07 11       21
   -------------------------------------------------------------------------------

   SARSOP finishing ...
     target precision reached
     target precision  : 0.000001
     precision reached : 0.000001

   -------------------------------------------------------------------------------
    Time   |#Trial |#Backup |LBound    |UBound    |Precision  |#Alphas |#Beliefs
   -------------------------------------------------------------------------------
    0       43      1393     19.3714    19.3714    9.22729e-07 5        21
   -------------------------------------------------------------------------------

   Writing out policy ...
     output file : temp-pomdp.policy

   [(TigerState(tiger-right), 0.5), (TigerState(tiger-left), 0.5)] listen tiger-left -1.0
   [(TigerState(tiger-left), 0.85), (TigerState(tiger-right), 0.15)] listen tiger-left -1.0
   [(TigerState(tiger-left), 0.9697986575573173), (TigerState(tiger-right), 0.03020134244268276)] open-right tiger-right 10.0
   ...

..
   PolicyGraph and AlphaVectorPolicy

PolicyGraph and AlphaVectorPolicy
---------------------------------

PolicyGraph and AlphaVectorPolicy extend the :py:mod:`~pomdp_py.framework.planner.Planner`
interface which means they have a :code:`plan` function that can be used to
output an action given an agent (using the agent's belief).

.. autoclass:: pomdp_py.utils.interfaces.conversion.PolicyGraph
   :members: construct, plan, update


.. autoclass:: pomdp_py.utils.interfaces.conversion.AlphaVectorPolicy
   :members: construct, value, plan
