Examples
========

In this document, we discuss some examples of using `pomdp_py` to define POMDP problems and solve them using :doc:`existing_solvers`. We will go over two examples: the simpler Tiger problem, and the more complicated but interesting Multi-Object Search (MOS) problem. The former represents how one could quickly implement a simple POMDP, and the latter represents how one may structure their project (as a convention) when using `pomdp_py` for more complicated domains.

We refer the reader to the following problems that have been implemented:

.. autosummary::
   :nosignatures:

   pomdp_problems.multi_object_search.problem
   pomdp_problems.tiger.tiger_problem
   pomdp_problems.rocksample.rocksample_problem



There are a number of problems that we hope to implement. There are even more examples `here <http://www.pomdp.org/examples/>`_.

* :doc:`problems/pomdp_problems.light_dark`
* :doc:`problems/pomdp_problems.load_unload`
* :doc:`problems/pomdp_problems.maze`
* :doc:`problems/pomdp_problems.tag`


In addition, the interfaces in `pomdp_py` are general enough to be extended to e.g. learning, multi-agent POMDPs; see :doc:`extensions`. `Contributions are always welcomed!`


Tiger
*****

This example describes how one could quickly implement a simple POMDP.

.. toctree::
   :maxdepth: 1
   
   examples.tiger

Multi-Object Search (MOS)
*************************

This example describes the convention of how one structures their project when using `pomdp_py` for more complicated domains.

.. toctree::
   :maxdepth: 1
   
   examples.mos   
