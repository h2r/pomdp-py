Changelog
=========

Best viewed on `the website <https://h2r.github.io/pomdp-py/html/changelog.html>`_.

Version 1.3.3 (07/25/2023)
-------------------------
* Bumped minimum Python requirement from 3.7 to 3.8 due to `EOL of 3.7 <https://devguide.python.org/versions/>`_.
* Fix :code:`cpdef -> cdef` to avoid installation failure after Cython 3.0.0 release (`pomdp-py#30 <https://github.com/h2r/pomdp-py/pull/30>`_).
* Added float_precision argument to to_pomdp_file (`pomdp-py#29 <https://github.com/h2r/pomdp-py/pull/29>`_)
* Add :code:`__init__` signature for Environment in comments to be visible in docs
* Fix :code:`s -> sp` in :py:mod:`~pomdp_py.algorithms.value_iteration.ValueIteration` (`pomdp-py#20 <https://github.com/h2r/pomdp-py/issues/20>`_)
* Allow updating rollout policy for :py:mod:`~pomdp_py.algorithms.po_uct.POUCT` and :py:mod:`~pomdp_py.algorithms.pomcp.POMCP`
* Fix in :code:`setup.py` so that wheel builds properly.
* Change set to list in :code:`pomdp_problems.tiger.tiger_problem.py` to tame error regarding :code:`random.sample` in Python 3.11.
* Minor bug fixes and documentation.


Version 1.3.2 (04/03/2022)
-------------------------
* Fix in :py:mod:`~pomdp_py.representations.distribution.histogram` (in :code:`__str__`):
  Print all of histogram as is instead of printing top 5 to avoid confusion.
* Improve documentation for `the tiger tutorial <https://h2r.github.io/pomdp-py/html/examples.tiger.html>`_;
  Specifically, clarified :py:mod:`~pomdp_py.framework.basics.PolicyModel`
  and gave a reference to :py:mod:`~pomdp_py.algorithms.po_uct.ActionPrior`.
* Built documentation with new Sphinx version (4.5.0)

Version 1.3.1 (11/03/2021)
-------------------------
* Bug fix in :py:mod:`~pomdp_py.representations.distribution.particles` (in :code:`add`)
* Added classes for tabular models :py:mod:`~pomdp_py.utils.templates.TabularTransitionModel`,
  :py:mod:`~pomdp_py.utils.templates.TabularObservationModel`,
  :py:mod:`~pomdp_py.utils.templates.TabularRewardModel`. See an example in `this Github Gist <https://gist.github.com/zkytony/51d43ee6818375434eb3b84a77a47a5c>`_ for defining and solving the CryingBaby domain, a small POMDP.

Version 1.3.0.1 (09/30/2021)
----------------------------
* Removed dependency on :code:`pygraphviz`;
* Added :code:`utils.debugging.Treedebugger`, which makes it easier to inspect the search tree.
  See :py:mod:`~pomdp_py.utils.debugging`.
* Added :code:`WeightedParticles`; Refactored :code:`Particles`. (:py:mod:`~pomdp_py.representations.distribution.particles`)
* Optionally show progress bar while simulating in POUCT/POMCP.
* Added a CLI interface to simplify running example domains, e.g. :code:`python -m pomdp_py -r tiger` runs Tiger.
* Can initialize :code:`Environment` with  :code:`BlackboxModel`.
* For the :code:`OOBelief` class in :code:`oopomdp.pyx`, now :code:`mpe` and :code:`random` can take an
  argument :code:`return_oostate` (default True), which returns a sampled state as type :code:`OOState`.
  This can be useful if you would like to inherit :code:`OOBelief` and return a state of
  your own type when implementing its :code:`mpe` and :code:`random` functions.
* Added :code:`__ne__` methods to framework classes.
* Reorganized :code:`util` by breaking it into different modules.
* Code refactoring: Remove unnecessary :code:`*args, **kwargs` in interface signature. **Backwards compatible**.
* Bug fix regarding hashing and pickling.
* Verified installation on Windows (TODO)


Version 1.2.4.6 (canceled)
--------------------------
* Fix :code:`setup.py` so that :code:`pip install -e .` works.

Version 1.2.4.5 (07/05/2021)
----------------------------
* Edit :code:`setup.py` file so that Cython modules in :code:`pomdp-py` can be :code:`cimport`ed.

(skipped versions due to attempting pypi release)

Version 1.2.4.1 (06/02/2021)
----------------------------
* Fix documentation (external solver examples).
* Update :code:`tiger_problem.py` to match documentation

Version 1.2.4 (06/01/2021)
--------------------------
* :code:`pomdp_py/algorithms/value_function.py`:
   * zero-probability observation should be skipped.
   * refactored so that :code:`value()` can take either a dict or a sequence of arguments.
* Available on `PyPI <https://pypi.org/project/pomdp-py/#history>`_
* :code:`.value` field of VNode is instead changed to be a property, computed by finding the maximum value of the children Q-Nodes.

Version 1.2.3 (03/22/2021)
--------------------------
* Bug fix in :code:`solvers.py` and :code:`conversion.py` (18fc58e0, cfc88e8d8)
* Bug fix in the MOS domain's observation model (719c2edf5)
* Linked `docker image <https://hub.docker.com/r/romainegele/pomdp>`_ in documentation `issue #13 <https://github.com/h2r/pomdp-py/issues)>`_.
* Updated documentations

Version 1.2.2.1 (01/25/2021)
----------------------------
* Updated documentation for external library interfacing. Added citation.

Version 1.2.2 (01/17/2021)
--------------------------
* Resolved `issue #10 <https://github.com/h2r/pomdp-py/issues/10>`_.
  Set value in V-Node to be the max among its children Q-Nodes.
  Initial V-Node value set to negative infinity.
* Avoid search tree building during rollout (thanks Jason)
* Documentation clarification about :code:`.sample` and :code:`.argmax` functions in the :code:`RewardModel`.
* Small pomdps (with enumerable state, action, observation spaces)
  defined in :code:`pomdp_py` can be converted to :code:`.pomdp` and :code:`.pomdpx` file formats.
* Added interfacing with `pomdp_solve <https://www.pomdp.org/code/>`_ and tested.
* Added interfacing with `sarsop <https://github.com/AdaCompNUS/sarsop>`_ and tested.
* Added :code:`utils/templates.py` that contains some convenient implementations of the POMDP interface.
* Bug fixes (in :code:`histogram.pyx`)


Version 1.2.1 (12/23/2020)
--------------------------
* Fixed preferred rollout and action prior implementation; Previously the initial visits and values were not applied.
* Fixed UCB1 value calculation when number of visits is 0; Previously a divide by zero error will be thrown. But it should have infinite value.
* Fixed another potential math domain error due to log(0) in UCB1 value calculation when initial number of visit set to 0.
* Fixed bug in particle belief update (minor type error)
* Simplified the Tiger example code, updated the Tiger example in documentation.
* Fixed bug in ValueIteration and verified in Tiger that it's able to
  differentiate differentiate between listen/stay actions when horizon = 3, but
  not so when horizon = 1 or 2. The same behavior is observed using the pomdp
  solver by `POMDP.org <https://www.pomdp.org/code/index.html>`_.
* Added an exact value function in :code:`pomdp_py.algorithms.value_function`. It is a simpler exact value iteration algorithm.
* Added Load/Unload domain `Pull request #9 <https://github.com/h2r/pomdp-py/pull/9>`_
* `Pull request #11 <https://github.com/h2r/pomdp-py/pull/11>`_

Pull Request #3 (08/01/2020)
----------------------------
* Added :code:`num_sims` parameter to POMCP/POUCT that allows specifying the number of simulations per planning step (Previously only :code:`planning_time` was available.
* Added cythonized versions of tiger and rocksample domains which are much faster.

Pull Request #1 (06/02/2020)
----------------------------
* Added continuous light-dark domain. A solver (BLQR) is attempted but not ready yet.
* Bug fix in 2D MOS domain rollout; action step size changeable
* Added Tag domain, tested with POUCT random rollout
* Documentation


Version 1.0 - 1.2 (02/08/2020 - 02/16/2020)
-------------------------------------------

* Rewritten with cleaner interfaces and Cython integration
* Refactored POMCP: Now it extends POUCT which does not assume particle belief representation.
* Included Tiger, RockSample, and a 2D multi-object search (MOS) domain.
* Using Sphinx Documentation


Version 0.0
-----------
* Implementation of POMCP and OO-POMDP
