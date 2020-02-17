Installation
============

We recommend using `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `pip <https://pip.pypa.io/en/stable/installing/>`_ to manage package versions and installation.

.. _install_pycy:

Install Python and Cython
-------------------------

0. Install Python 3.7+ from `official website <https://www.python.org/downloads/>`_.

1. Install Cython.::

    pip install Cython

.. _get_pomdp_py:

Install pomdp_py
----------------

1. **First**, download `pomdp_py` latest release `on github <https://github.com/h2r/pomdp-py/releases>`_, or clone the repository by::

    git clone https://github.com/h2r/pomdp-py.git
     
2. **Next**, go to the package root directory (where :code:`setup.py` is located). Run::
    
    pip install -e .

   This will build and install `pomdp_py` package. This will build :code:`.so` files and copy them to the python source directory.  When you make changes to :code:`.pyx` or :code:`.pyd` files, run the following to rebuild those :code:`.so` libraries, so that the python imports can get those changes you made::

     make build
     
3. **Finally**, verify that Tiger problem and RockSample problem work::
   
     python pomdp_problems/tiger/tiger_problem.py
     python pomdp_problems/rocksample/rocksample_problem.py
   
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
   
