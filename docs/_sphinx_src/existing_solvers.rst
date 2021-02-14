Existing POMDP Solvers
======================

The library currently contains an implementation of `POMCP`, `POUCT`, and basic `ValueIteration`.

.. autosummary::
   :nosignatures:

   ~pomdp_py.algorithms.po_rollout
   ~pomdp_py.algorithms.po_uct
   ~pomdp_py.algorithms.pomcp
   ~pomdp_py.algorithms.value_iteration.ValueIteration


The library also currently interfaces with `pomdp-solve <https://www.pomdp.org/code/>`_, developed by Anthony R. Cassandra, and `sarsop <https://github.com/AdaCompNUS/sarsop>`_, developed by NUS. See :doc:`examples.external_solvers` for details and examples.

.. autosummary::
   :nosignatures:

   ~pomdp_py.utils.interfaces.solvers.sarsop
   ~pomdp_py.utils.interfaces.solvers.vi_pruning

.. note::

   A pomdp_py :py:mod:`~pomdp_py.framework.basics.Agent` with enumerable state :math:`S`, action :math:`A`, and observation spaces :math:`\Omega`, with explicitly defined probability for its models (:math:`T,O,R`) can be directly used as input to the above functions that interface with the solvers' binaries programs.
