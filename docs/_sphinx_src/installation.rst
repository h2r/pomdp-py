Installation
============

0. Install Python 3.7+ from `official website <https://www.python.org/downloads/>`_.

1. Install Cython.::

    pip install Cython

2. Download `pomdp_py` latest release `on github <https://github.com/h2r/pomdp-py/releases>`_.

3. Go to the package root directory. Install as a developer::
    
    pip install -e .

   This will install `pomdp_py` package and compile the Cython code in place.

4. Verify that Tiger problem and RockSample problem work::

    python problems/tiger/tiger_problem.py
    python problems/rocksample/rocksample_problem.py

   For the Tiger problem, you should see output like
   
   .. code-block:: text
   
    ** Testing value iteration **
    ==== Step 1 ====
    True state: tiger-left
    Belief: [(State(tiger-right), 0.5), (State(tiger-left), 0.5)]
    Action: listen
    Reward: -1
    >> Observation: tiger-left
    ...

  There will be plots that visualize the MCTS trees displayed,

  For the RockSample problem, you should see something like::

    *** Testing POMCP ***
    ==== Step 1 ====
    Particle reinvigoration for 66 particles
    True state: State((0, 4) | ('bad', 'good', 'bad', 'good', 'good') | False)
    Action: sample
    Observation: None
    Reward: 0.0
    Reward (Cumulative): 0.0
    Reward (Cumulative Discounted): 0.0
    __num_sims__: 1217
    World:
    
    ______ID______
    .....>
    4....>
    ..210>
    .3...>
    R....>
    _____G/B_____
    .....>
    $....>
    ..x$x>
    .$...>
    R....>
