
pomdp_py Documentation
======================


Overview
--------
`pomdp_py <https://github.com/h2r/pomdp-py>`_ is a **general purpose POMDP library** written in Python and Cython. It features simple and comprehensive interfaces to describe POMDP or MDP problems. Originally written to support POMDP planning research, the interfaces also allow extensions to model-free or model-based learning in (PO)MDPs, multi-agent POMDP planning/learning, and task transfer or transfer learning.

POMDP stands for **P**\ artially **O**\ bservable **M**\ arkov **D**\ ecision **P**\ rocess :cite:`kaelbling1998planning`.

The code is available `on github <https://github.com/h2r/pomdp-py>`_.


Getting Started
---------------
* :doc:`installation`
* :doc:`examples` (:doc:`examples.tiger`, :doc:`examples.mos`).
  The Tiger example represents how one could quickly implement a simple POMDP; The Multi-Object Search domain represents how one may structure their project (as a convention) when using `pomdp_py` for more complicated domains.
  
* :doc:`design_principles`
* :doc:`existing_solvers`
* :doc:`extensions`
* :doc:`simple_rl_integration`

Further
-------
* :doc:`design_principles`
* :doc:`existing_solvers`
* :doc:`extensions`
* :doc:`other_libraries`  

.. toctree::
   :maxdepth: 2
   :caption: Overview              
   :hidden:

   installation
   examples
   design_principles
   existing_solvers
   extensions         
 

API References
--------------
.. toctree::
   :maxdepth: 3
   :caption: API References

   api/modules
   problems/modules   

Tools
-----
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. bibliography:: refs.bib
   :filter: docname in docnames
   :style: unsrt

.. image:: images/brown_logo.png
   :target: http://bigai.cs.brown.edu/
   :scale: 25 %
   :alt: Brown University AI
   :align: center
