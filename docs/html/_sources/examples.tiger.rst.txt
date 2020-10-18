Tiger Problem
*************

This is a classic POMDP problem, introduced in :cite:`kaelbling1998planning`. The description of the tiger problem is as follows: (Quote from `POMDP:
Introduction to Partially Observable Markov Decision Processes
<https://cran.r-project.org/web/packages/pomdp/vignettes/POMDP.pdf>`_ by
Kamalzadeh and Hahsler ):

`A tiger is put with equal probability behind one
of two doors, while treasure is put behind the other one.
You are standing in front of the two closed doors and
need to decide which one to open. If you open the door
with the tiger, you will get hurt (negative reward).
But if you open the door with treasure, you receive
a positive reward. Instead of opening a door right away,
you also have the option to wait and listen for tiger noises. But
listening is neither free nor entirely accurate. You might hear the
tiger behind the left door while it is actually behind the right
door and vice versa.`

Tiger is a simple POMDP with only 2 states, 2 actions, and 2 observations.
In pomdp_py, to define and solve a POMDP:

1. :ref:`define-the-domain`
2. :ref:`define-the-models`
3. :ref:`instantiate`
4. :ref:`solve`

.. note::

   For a simple POMDP like Tiger, it is encouraged to place the code for all components (e.g. state, action, observation and models) under the same Python module (i.e. the same :code:`.py` file).

.. _define-the-domain:

Define the domain
-----------------

We start by defining the domain (:math:`S, A, O`). In `pomdp_py`, this is
equivalent as defining three classes that inherit
:py:mod:`~pomdp_py.framework.basics.State`,
:py:mod:`~pomdp_py.framework.basics.Action`,
:py:mod:`~pomdp_py.framework.basics.Observation`
(see :py:mod:`~pomdp_py.framework.basics`).

.. code-block:: python

    class State(pomdp_py.State):
        def __init__(self, name):
            if name != "tiger-left" and name != "tiger-right":
                raise ValueError("Invalid state: %s" % name)
            self.name = name
        # ... __hash__, __eq__ should be implemented

.. code-block:: python

    class Action(pomdp_py.Action):
        def __init__(self, name):
            if name != "open-left" and name != "open-right"\
               and name != "listen":
                raise ValueError("Invalid action: %s" % name)
            self.name = name
        # ... __hash__, __eq__ should be implemented

.. code-block:: python

    class Observation(pomdp_py.Observation):
        def __init__(self, name):
            if name != "tiger-left" and name != "tiger-right":
                raise ValueError("Invalid action: %s" % name)
            self.name = name
        # ... __hash__, __eq__ should be implemented

`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#State>`_

.. _define-the-models:

Define the models
------------------

Next, we define the models (:math:`T, O, R, \pi`). In `pomdp_py`, this is
equivalent as defining classes that inherit
:py:mod:`~pomdp_py.framework.basics.ObservationModel`,
:py:mod:`~pomdp_py.framework.basics.TransitionModel`,
:py:mod:`~pomdp_py.framework.basics.RewardModel`,
:py:mod:`~pomdp_py.framework.basics.PolicyModel`    (see
:py:mod:`~pomdp_py.framework.basics`).

.. note::

   `pomdp_py` also provides an interface for :py:mod:`~pomdp_py.framework.basics.BlackboxModel`.


We begin with the :py:mod:`~pomdp_py.framework.basics.ObservationModel`. In Tiger, when the agent takes the listen action, it observes which side the tiger is with some noise. Implementing such a model in pomdp_py boils down to implementing a generative model with an optional :code:`probability` function that you can implement when, for example, you need to perform exact belief update. One way of implementing this is as follows. Note that our model inherits the pomdp_py's :py:mod:`~pomdp_py.framework.basics.ObservationModel` interface.

.. code-block:: python

  class ObservationModel(pomdp_py.ObservationModel):
      def __init__(self, noise=0.15):
          self.noise = noise

      def probability(self, observation, next_state, action):
          if action.name == "listen":
              # heard the correct growl
              if observation.name == next_state.name:
                  return 1.0 - self.noise
              else:
                  return self.noise
          else:
              return 0.5

      def sample(self, next_state, action):
          if action.name == "listen":
              thresh = 1.0 - self.noise
          else:
              thresh = 0.5

          if random.uniform(0,1) < thresh:
              return Observation(next_state.name)
          else:
              return Observation(next_state.other().name)

      def get_all_observations(self):
          """Only need to implement this if you're using
          a solver that needs to enumerate over the observation
          space (e.g. value iteration)"""
          return [Observation(s)
                  for s in {"tiger-left", "tiger-right"}]
`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#ObservationModel>`_

The :py:mod:`~pomdp_py.framework.basics.TransitionModel` is deterministic. Similarly, we implement the :code:`sample` and :code:`probability` functions in the interface for this generative model:


.. code-block:: python

  class TransitionModel(pomdp_py.TransitionModel):
      def probability(self, next_state, state, action):
          """According to problem spec, the world resets once
          action is open-left/open-right. Otherwise, it
          stays the same"""
          if action.name.startswith("open"):
              return 0.5
          else:
              if next_state.name == state.name:
                  return 1.0 - 1e-9
              else:
                  return 1e-9

      def sample(self, state, action):
          if action.name.startswith("open"):
              return random.choice(self.get_all_states())
          else:
              return State(state.name)

      def get_all_states(self):
          """Only need to implement this if you're using
          a solver that needs to enumerate over the
          observation space (e.g. value iteration)"""
          return [State(s) for s in {"tiger-left", "tiger-right"}]

`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#TransitionModel>`_


Since the Tiger domain is small, the transition and observation probabilities can be easily specified by a table (a dictionary in Python), which is similar to specifying POMDPs using POMDP file formats. However, pomdp_py allows more flexible way of implementing these models which can be intractable to enumerate (e.g. continuous).

Next, we define the :py:mod:`~pomdp_py.framework.basics.PolicyModel`. The job of
a PolicyModel is to (1) determine the set of actions that the robot can take at
given state (and/or history); (2) sample an action from this set according to
some probability distribution. This allows extensions to policy models that have
a prior over actions. The idea of preference over actions have been used in
several existing work :cite:`silver2010monte` :cite:`abel2015goal`
:cite:`xiao_icra_2019`.  Without prior knowledge of action preference, the
PolicyModel can simply sample actions from the set uniformly. Typically, we
would like to start without (usually human-engineered) prior knowledge over
actions, because it sort of guides the planner and we are not sure if this
guidance based on heuristics is actually optimal. So caution must be used.

In the Tiger problem, we just define a simple PolicyModel as follows.  We choose
not to implement the :code:`probability` and :code:`argmax` functions because we
don't really use them for planning; The PolicyModel in this case can do (1)
and (2) without those two functions. But in general, the PolicyModel could
be learned, or the action space is large so a probability distribution over
it becomes important.

.. code-block:: python

   class PolicyModel(pomdp_py.RandomRollout):
       """A simple policy model with uniform prior over a
          small, finite action space"""
       ACTIONS = {Action(s)
                 for s in {"open-left", "open-right", "listen"}}

       def sample(self, state, **kwargs):
           return self.get_all_actions().random()

       def get_all_actions(self, **kwargs):
           return PolicyModel.ACTIONS

`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#PolicyModel>`_

Finally, we define the :py:mod:`~pomdp_py.framework.basics.RewardModel`.
It is straightforward according to the problem description. In this case,
(and very commonly), the reward function is deterministic. We can implement
it as follows. The interface for reward model does allow stochastic rewards.

.. code-block:: python

  class RewardModel(pomdp_py.RewardModel):
      def _reward_func(self, state, action):
          if action.name == "open-left":
              if state.name == "tiger-right":
                  return 10
              else:
                  return -100
          elif action.name == "open-right":
              if state.name == "tiger-left":
                  return 10
              else:
                  return -100
          else: # listen
              return -1

      def sample(self, state, action, next_state):
          # deterministic
          return self._reward_func(state, action)

`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#RewardModel>`_


Define the POMDP
----------------

With the models that we have defined, it is simple to define a POMDP for the Tiger
problem; To do this, we need to define :py:mod:`~pomdp_py.framework.basics.Agent`,
and :py:mod:`~pomdp_py.framework.basics.Environment`. Note that you could just construct an agent and an environment and still be able to plan actions and simulate the environment.
This class is mostly just for code organization and is entirely optional.

.. code-block:: python

    class TigerProblem(pomdp_py.POMDP):

        def __init__(self, obs_noise, init_true_state, init_belief):
            """init_belief is a Distribution."""
            agent = pomdp_py.Agent(init_belief,
                                   PolicyModel(),
                                   TransitionModel(),
                                   ObservationModel(obs_noise),
                                   RewardModel())
            env = pomdp_py.Environment(init_true_state,
                                       TransitionModel(),
                                       RewardModel())
            super().__init__(agent, env, name="TigerProblem")

`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#TigerProblem>`_

Notice that :code:`init_true_state` and :code:`init_belief` need to be provided.
The process of creating them is described in more detail in the next section.

.. note::

   It is entirely optional to define a `Problem` class (like
   :code:`TigerProblem`) that extends the
   :py:mod:`pomdp_py.framework.basics.POMDP` class in order to use a
   :py:mod:`pomdp_py.framework.planner.Planner` to solve a POMDP; Only the
   `Agent` and the `Environment` are needed. The POMDP class sometimes can
   organize the parameters that need to be passed into the constructors of
   `Agent` and `Environment`. For complicated problems, specific `Agent` and
   `Environment` classes are written that inherit
   :py:mod:`pomdp_py.framework.basics.Agent` and
   :py:mod:`pomdp_py.framework.basics.Environment`.


.. _instantiate:

Instantiate the POMDP
-----------------------

Now we have a definition of the Tiger problem. Now, we need to `instantiate`
a problem by providing `parameters` for the models,
the `initial state` of the environment, and the `initial belief` of the agent.

In Tiger, the model parameters are basically the probabilities for :math:`T`
and :math:`O`, which have been described above (see :ref:`define-the-models`).

We can create a random initial state and a uniform belief as follows:

.. code-block:: python

   init_true_state = random.choice([State("tiger-left"),
                                    State("tiger-right")])
   init_belief = pomdp_py.Histogram({State("tiger-left"): 0.5,
                                     State("tiger-right"): 0.5})

Then, we can create an instance of the Tiger problem with the standard noise of 0.15:

.. code-block:: python

   tiger_problem = TigerProblem(0.15, init_true_state, init_belief)


`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#main>`_


.. _solve:

Solve the POMDP instance
--------------------------

To solve a POMDP with `pomdp_py`, here are the basic steps:

1. Create a planner (:py:mod:`~pomdp_py.framework.planner.Planner`)

2. Agent plans an action :math:`a_t`.

3. Environment state transitions :math:`s_t \rightarrow s_{t+1}`
   according to its transition model.

4. Agent receives an observation :math:`o_t` and reward :math:`r_t` from the environment.

5. Agent updates history and belief :math:`h_t,b_t \rightarrow h_{t+1},b_{t+1}` where :math:`h_{t+1} = h_t \cup (a_t, o_t)`.

   * This could be done either by updating the :code:`belief` of
     an agent directly, or through an update of the planner. More
     specifically, if the planner is :py:mod:`~pomdp_py.algorithms.pomcp.POMCP`, updating the planner
     will result in the agent belief update as well. But for
     :py:mod:`~pomdp_py.algorithms.pomcp.POUCT` or :py:mod:`~pomdp_py.algorithms.pomcp.ValueIteration`, the agent belief needs to be updated explicitly.

6. Unless termination condition is reached, repeat steps 2-6.

For the Tiger problem, we implemented this procedure as follows:

.. code-block:: python

    # Step 1; in main()
    # creating planners
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                           planning_time=.5, exploration_const=110,
                           rollout_policy=tiger_problem.agent.policy_model)
    pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                           planning_time=.5, exploration_const=110,
                           rollout_policy=tiger_problem.agent.policy_model)
    ...  # call test_planner() for steps 2-6.

    # Steps 2-6; called in main()
    def test_planner(tiger_problem, planner, nsteps=3):
       """Runs the action-feedback loop of Tiger problem POMDP"""
        for i in range(nsteps):  # Step 6
            # Step 2
            action = planner.plan(tiger_problem.agent)
            print("==== Step %d ====" % (i+1))
            print("True state: %s" % tiger_problem.env.state)
            print("Belief: %s" % str(tiger_problem.agent.cur_belief))
            print("Action: %s" % str(action))
            # Step 3; no transition since actions in Tiger problem
            # does not change environment state (i.e. tiger location).
            print("Reward: %s" % str(tiger_problem.env.reward_model.sample(tiger_problem.env.state, action, None)))

            # Step 4
            # Let's create some simulated real observation; Update the belief
            # Creating true observation for sanity checking solver behavior.
            # In general, this observation should be sampled from agent's observation model.
                real_observation = Observation(tiger_problem.env.state.name)
            print(">> Observation: %s" % real_observation)

            # Step 5
            tiger_problem.agent.update_history(action, real_observation)
            planner.update(tiger_problem.agent, action, real_observation)
            if isinstance(planner, pomdp_py.POUCT):
                print("Num sims: %d" % planner.last_num_sims)
            if isinstance(tiger_problem.agent.cur_belief, pomdp_py.Histogram):
                new_belief = pomdp_py.update_histogram_belief(tiger_problem.agent.cur_belief,
                                                              action, real_observation,
                                                              tiger_problem.agent.observation_model,
                                                              tiger_problem.agent.transition_model)
                tiger_problem.agent.set_belief(new_belief)

`[source] <_modules/pomdp_problems/tiger/tiger_problem.html#test_planner>`_

.. _summary:

Summary
-------

In short, to use `pomdp_py` to define a POMDP problem and solve an instance of the problem,

1. :ref:`define-the-domain`
2. :ref:`define-the-models`
3. :ref:`instantiate`
4. :ref:`solve`

.. bibliography:: refs.bib
   :filter: docname in docnames
   :style: unsrt
