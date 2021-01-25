### Version 1.2.2.1
* Updated documentation for external library interfacing. Added citation.

### Version 1.2.2
* Resolved [issue #10](https://github.com/h2r/pomdp-py/issues/10).
  Set value in V-Node to be the max among its children Q-Nodes.
  Initial V-Node value set to negative infinity.
* Avoid search tree building during rollout (thanks Jason)
* Documentation clarification about `.sample` and `.argmax` functions in the `RewardModel`.
* Small pomdps (with enumerable state, action, observation spaces)
  defined in `pomdp_py` can be converted to `.pomdp` and `.pomdpx` file formats.
* Added interfacing with [pomdp_solve](https://www.pomdp.org/code/) and tested.
* Added interfacing with [sarsop](https://github.com/AdaCompNUS/sarsop) and tested.
* Added `utils/templates.py` that contains some convenient implementations of the POMDP interface.
* Bug fixes (in `histogram.pyx`)


### Version 1.2.1

* Fixed preferred rollout and action prior implementation; Previously the initial visits and values were not applied.
* Fixed UCB1 value calculation when number of visits is 0; Previously a divide by zero error will be thrown. But it should have infinite value.
* Fixed another potential math domain error due to log(0) in UCB1 value calculation when initial number of visit set to 0.
* Fixed bug in particle belief update (minor type error)
* Simplified the Tiger example code, updated the Tiger example in documentation.
* Fixed bug in ValueIteration and verified in Tiger that it's able to
  differentiate differentiate between listen/stay actions when horizon = 3, but
  not so when horizon = 1 or 2. The same behavior is observed using the pomdp
  solver by [POMDP.org](https://www.pomdp.org/code/index.html).
* Added an exact value function in `pomdp_py.algorithms.value_function`. It is a simpler exact value iteration algorithm.
* Added Load/Unload domain [Pull request #9](https://github.com/h2r/pomdp-py/pull/9)
* [Pull request #11](https://github.com/h2r/pomdp-py/pull/11)

### Pull Request #3 (08/01/2020)

* Added `num_sims` parameter to POMCP/POUCT that allows specifying the number of simulations per planning step (Previously only `planning_time` was available.
* Added cythonized versions of tiger and rocksample domains which are much faster.

### Pull Request #1 (06/02/2020)

* Added continuous light-dark domain. A solver (BLQR) is attempted but not ready yet.
* Bug fix in 2D MOS domain rollout; action step size changeable
* Added Tag domain, tested with POUCT random rollout
* Documentation


### Version 1.0 - 1.2

* Rewritten with cleaner interfaces and Cython integration
* Refactored POMCP: Now it extends POUCT which does not assume particle belief representation.
* Included Tiger, RockSample, and a 2D multi-object search (MOS) domain.
* Using Sphinx Documentation


### Version 0.0

* Implementation of POMCP and OO-POMDP
