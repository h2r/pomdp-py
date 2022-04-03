Preference-based Action Prior
*****************************

The code below is a minimum example of defining a
:py:mod:`~pomdp_py.framework.basics.PolicyModel`
that supports a rollout policy based on preference-based action prior :cite:`silver2010monte`.
The action prior is specified through the
:py:mod:`~pomdp_py.algorithms.po_uct.ActionPrior` object,
which returns a set of preferred actions given a state (and/or history).

.. code-block:: python

    import random
    from pomdp_py import RolloutPolicy, ActionPrior

    class PolicyModel(RolloutPolicy):
        def __init__(self, action_prior=None):
            """
            action_prior is an object of type ActionPrior
            that implements that get_preferred_actions function.
            """
            self.action_prior = action_prior

        def sample(self, state):
            return random.sample(
                self.get_all_actions(state=state), 1)[0]

        def get_all_actions(self, state, history=None):
            raise NotImplementedError

        def rollout(self, state, history=None):
            if self.action_prior is not None:
                preferences =\
                    self.action_prior\
                        .get_preferred_actions(state, history)
                if len(preferences) > 0:
                    return random.sample(preferences, 1)[0][0]
                else:
                    return self.sample(state)
            else:
                return self.sample(state)

Note that the notion of "action prior" here is narrow; It
follows the original POMCP paper :cite:`silver2010monte`.
In general, you could express a prior over the action distribution
explicitly through the :code:`sample` and :code:`rollout` function in
:py:mod:`~pomdp_py.framework.basics.PolicyModel`. Refer to the `Tiger <https://h2r.github.io/pomdp-py/html/examples.tiger.html#:~:text=e.g.%20continuous).-,Next,-%2C%20we%20define%20the>`_
tutorial for more details (the paragraph on PolicyModel).

As described in :cite:`silver2010monte`, you could choose to set an initial visit count and initial value corresponding
to a preferred action; To take this into account during POMDP planning using POUCT or POMCP,
you need to supply the :py:mod:`~pomdp_py.algorithms.po_uct.ActionPrior` object
when you initialize the :py:mod:`~pomdp_py.algorithms.po_uct.POUCT`
or :py:mod:`~pomdp_py.algorithms.pomcp.POMCP` objects through the :code:`action_prior` argument.
