Examples
========

In this document, we discuss some examples of using `pomdp_py` to define POMDP problems and solve them using :doc:`existing_solvers`. We will go over two examples: the simpler Tiger problem, and the more complicated but interesting Multi-Object Search (MOS) problem. The former represents how one could quickly implement a simple POMDP, and the latter represents how one may structure their project (as a convention) when using `pomdp_py` for more complicated domains.

We refer the reader to the following problems that have been implemented:

.. autosummary::
   :nosignatures:

   problems.multi_object_search.problem
   problems.tiger.tiger_problem
   problems.rocksample.rocksample_problem



There are a number of problems that we hope to implement. There are even more examples `here <http://www.pomdp.org/examples/>`_.

* :doc:`problems/problems.light_dark`
* :doc:`problems/problems.load_unload`
* :doc:`problems/problems.maze`
* :doc:`problems/problems.tag`


In addition, the interfaces in `pomdp_py` are general enough to be extended to e.g. learning, multi-agent POMDPs; see :doc:`extensions`. `Contributions are always welcomed!`


Tiger
*****

.. toctree::
   :maxdepth: 1
   
   examples.tiger

Multi-Object Search (MOS)
*************************

.. toctree::
   :maxdepth: 1
   
   examples.mos   

..
   `problems.tiger.tiger_problem.State`


