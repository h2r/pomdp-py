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


